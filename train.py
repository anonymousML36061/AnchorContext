import argparse
import torch
import os
import math
from tqdm import tqdm
from datetime import timedelta
import random
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets, Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForTokenClassification, AutoConfig
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM
import transformers
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from accelerate.utils import (
    InitProcessGroupKwargs,
    set_seed,
    DummyOptim,
    DummyScheduler,
)


from anchor_context import (
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
    prepare_dataloader,
)

import json
from typing import List, Dict, Any, Union
from torch.nn.attention.flex_attention import create_block_mask
from deepspeed import get_accelerator
from utils import get_checkpoint_list, clean_checkpoint_folder, setup_debugpy

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

# Define global variables
IGNORE_INDEX=-100
PLACEHOLDER=-2024
BOS_TOKEN_ID = 1
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"


# get current file dir
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

def model_shortname(model_name):
    if 'llama2' in model_name.lower()  or 'llama-2' in model_name.lower() or 'llama_2' in model_name.lower():
        return 'llama2'
    elif 'llama3' in model_name.lower()  or 'llama-3' in model_name.lower() or 'llama_3' in model_name.lower():
        return 'llama3'
    elif 'mistral' in model_name.lower():
        return 'mistral'
    elif 'qwen' in model_name.lower():
        return 'qwen'
    else:
        raise ValueError("model_name not supported")
    


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def process_tokenized_data(examples, scaled_max_position_embeddings, anchor_token, attn_engine, interleaved_chunks=False):
    """
    Processes tokenized data by adjusting sequences to fit a specified maximum length,
    handling anchor tokens based on the attention engine, and packing input IDs and
    end-of-sequence indices for model input.

    Parameters:
    - examples: A dictionary with keys 'input_ids' and 'source'.
      - 'input_ids': List of lists, where each sublist contains token IDs of a sequence.
      - 'source': List of lists, where each sublist contains dicts with 'end' indices.
    - scaled_max_position_embeddings: The maximum length for processed sequences.
    - anchor_token: List of token IDs representing the anchor token(s).
    - attn_engine: String indicating the attention mechanism ('flex' or 'flash').

    Returns:
    - model_inputs: A dictionary with keys 'input_ids' and 'source_eos', containing
      the processed input IDs and end-of-sequence indices.
    """
    packed_input_ids = []
    packed_eos_index = []
    anchor_token_len = len(anchor_token)

    # Adjust maximum position embeddings based on the attention engine
    if attn_engine == 'flex':
        scaled_max_position_embeddings -= anchor_token_len
    elif attn_engine != 'flash':
        raise ValueError(f"Unsupported attn_engine: {attn_engine}")

    # Process each sequence in the examples
    for idx, input_ids in enumerate(examples['input_ids']):
        eos_idx_list = [s_info["end"] for s_info in examples['source'][idx]]
        prev_eos_idx = 0
        tmp_input_ids = []
        temp_packed_eos_index = []

        for eos_idx in eos_idx_list:
            # Extract the current segment
            segment = input_ids[prev_eos_idx:eos_idx]
            prev_eos_idx = eos_idx  # Update for the next iteration

            # Skip empty or too short segments
            if len(segment) <= 2:
                continue

            # Determine whether to skip or add the anchor token
            if attn_engine == 'flex':
                # Skip the anchor token if present at the start of the segment
                if segment[:anchor_token_len] == anchor_token:
                    segment = segment[anchor_token_len:]
            elif attn_engine == 'flash':
                # Ensure the segment starts with the anchor token
                if segment[:anchor_token_len] != anchor_token:
                    segment = anchor_token + segment

            # Check if adding the segment exceeds the maximum length
            if len(tmp_input_ids) + len(segment) > scaled_max_position_embeddings:
                # Trim the segment to fit the remaining space
                remaining_space = scaled_max_position_embeddings - len(tmp_input_ids)
                if remaining_space <= 0:
                    break  # No more space left in the sequence
                segment = segment[:remaining_space]

                # Adjust the EOS index accordingly
                temp_packed_eos_index.append(len(tmp_input_ids) + len(segment))
                tmp_input_ids.extend(segment)
                break  # Sequence is full after adding this segment
            else:
                # Update EOS index before adding the segment
                temp_packed_eos_index.append(len(tmp_input_ids) + len(segment))
                tmp_input_ids.extend(segment)

        assert len(tmp_input_ids) == temp_packed_eos_index[-1], f"Length mismatch: {len(tmp_input_ids)} vs {temp_packed_eos_index[-1]}"
        assert len(tmp_input_ids) >= scaled_max_position_embeddings
        # Truncate in case of any overflow
        tmp_input_ids = tmp_input_ids[:scaled_max_position_embeddings]
        # Ensure EOS indices do not exceed the max length
        temp_packed_eos_index = [idx for idx in temp_packed_eos_index if idx < scaled_max_position_embeddings]
        temp_packed_eos_index.append(scaled_max_position_embeddings)

        # Append the processed sequence and EOS indices to the lists
        packed_input_ids.append(tmp_input_ids)
        packed_eos_index.append(temp_packed_eos_index)

    model_inputs = {"input_ids": packed_input_ids, "source_eos": packed_eos_index}
    
    if interleaved_chunks:
        assert attn_engine == 'flex', "interleaved_chunks only works with flex attention"
        model_inputs = split_shuffle_join(model_inputs)
    else:
        model_inputs = add_source_doc_ids(model_inputs)

    if attn_engine == 'flex':
        model_inputs = add_anchor_token_to_inputs(model_inputs, anchor_token)

    return model_inputs


def process_json_data(examples, tokenizer, scaled_max_position_embeddings, anchor_token, attn_engine):
    inputs = examples["text"]
    raw_model_inputs = tokenizer(inputs, padding=False)['input_ids']

    packed_input_ids = []
    packed_eos_index = []
    tmp_input_ids = []
    tmp_eos_index = []
    for i in range(len(raw_model_inputs)):
        input_ids = raw_model_inputs[i]

        if len(input_ids) <= 2:  # Skip empty strings
            continue

        # Handle input_ids longer than scaled_max_position_embeddings
        while len(input_ids) > 0:
            # if input_ids[0] == anchor_token[0]:
            available_space = scaled_max_position_embeddings - len(tmp_input_ids) # - (1 if attn_engine == 'flash' else 0)
            chunk_size = min(len(input_ids), available_space)
            chunk = input_ids[:chunk_size]
            input_ids = input_ids[chunk_size:]  # Remaining tokens

            prev_length = tmp_eos_index[-1] if tmp_input_ids else 0
            if attn_engine == 'flash':
                _to_add_seq = anchor_token + chunk if chunk[0] != anchor_token[0] else chunk
            else:
                _to_add_seq = chunk if chunk[0] != anchor_token[0] else chunk[len(anchor_token):]
            tmp_input_ids += _to_add_seq
            tmp_eos_index.append(prev_length + len(_to_add_seq))

            if len(tmp_input_ids) >= scaled_max_position_embeddings:
                packed_input_ids.append(tmp_input_ids[:scaled_max_position_embeddings])
                # Adjust tmp_eos_index if necessary
                if tmp_eos_index:
                    # Remove indices that are equal to or larger than the maximum length
                    tmp_eos_index = [idx for idx in tmp_eos_index if idx < scaled_max_position_embeddings]
                    # Ensure the last EOS index corresponds to the maximum length
                    if tmp_eos_index and tmp_eos_index[-1] != scaled_max_position_embeddings:
                        tmp_eos_index.append(scaled_max_position_embeddings)
                else: # If there are no EOS indices, add the maximum length as the EOS index
                    tmp_eos_index = [scaled_max_position_embeddings]
                packed_eos_index.append(tmp_eos_index)
            
                tmp_input_ids = []
                tmp_eos_index = []

    model_inputs = {"input_ids": packed_input_ids, "source_eos": packed_eos_index}
    model_inputs = add_source_doc_ids(model_inputs)

    if attn_engine == 'flex':
        model_inputs = add_anchor_token_to_inputs(model_inputs, anchor_token)
    return model_inputs


def preprocess_raw_data(examples, data_format, tokenizer, scaled_max_position_embeddings, anchor_token, attn_engine, interleaved_chunks=False):
    if data_format == "raw_json":
        return process_json_data(examples, tokenizer, scaled_max_position_embeddings, anchor_token, attn_engine)
    else:
        return process_tokenized_data(examples, scaled_max_position_embeddings, anchor_token, attn_engine)
    

def add_anchor_token_to_inputs(model_inputs, anchor_token):
    packed_input_ids = model_inputs['input_ids']
    packed_eos_index = model_inputs['source_eos']
    paced_doc_ids = model_inputs['source_doc_ids']
    final_input_ids, final_eos_indices, final_doc_ids = [], [], []
    for _seq_id in range(len(packed_input_ids)):
        input_ids = packed_input_ids[_seq_id]
        # assert anchor not appear in input_ids
        assert anchor_token[0] not in input_ids
        eos_indices = packed_eos_index[_seq_id]
        doc_ids = paced_doc_ids[_seq_id]
        final_input_ids.append(anchor_token + input_ids)
        final_eos_indices.append([len(anchor_token)] + [len(anchor_token) + eos_idx for eos_idx in eos_indices])
        # final_doc_ids.append([0] + [1 + doc_id for doc_id in doc_ids])
        final_doc_ids.append([1 + doc_id for doc_id in doc_ids])

    model_inputs['input_ids'] = final_input_ids
    model_inputs['source_eos'] = final_eos_indices
    model_inputs['source_doc_ids'] = final_doc_ids
    return model_inputs


def add_source_doc_ids(model_inputs):
    packed_input_ids = model_inputs['input_ids']
    packed_eos_index = model_inputs['source_eos']
    final_input_ids, final_eos_indices, final_doc_ids = [], [], []
    for _seq_id in range(len(packed_input_ids)):
        input_ids = packed_input_ids[_seq_id]
        eos_indices = packed_eos_index[_seq_id]

        final_input_ids.append(input_ids)
        final_eos_indices.append(eos_indices)
        final_doc_ids.append( list(range(len(eos_indices))) )

    # Update model_inputs
    model_inputs['input_ids'] = final_input_ids
    model_inputs['source_eos'] = final_eos_indices
    model_inputs['source_doc_ids'] = final_doc_ids

    return model_inputs


def split_shuffle_join(model_inputs, min_seq_len=64, max_seq_len=2048, interval=12):
    packed_input_ids = model_inputs['input_ids']
    packed_eos_index = model_inputs['source_eos']
    final_input_ids, final_eos_indices, final_doc_ids = [], [], []
    
    for _seq_id in range(len(packed_input_ids)):
        input_ids = packed_input_ids[_seq_id]
        eos_indices = packed_eos_index[_seq_id]
        if eos_indices[0] != 0:
            eos_indices = [0] + eos_indices
        all_chunks = []
        for doc_id in range(len(eos_indices)-1):
            start = eos_indices[doc_id]
            end = eos_indices[doc_id + 1]
            doc_ids = input_ids[start:end]
            
            doc_len = len(doc_ids)
            # Handle cases where the sequence is very short or very long
            if doc_len < min_seq_len or doc_len > max_seq_len: # not split
                splits = []
            else:
                assert doc_len > 30
                # Decide whether to split into 2 or 3 parts
                num_parts = random.choice([2, 3])
                # Avoid splitting at the very beginning or end
                min_idx = int(0.2 * doc_len)
                max_idx = int(0.8 * doc_len)
                if num_parts == 2:
                    splits = random.sample(range(min_idx, max_idx), 1)
                else:
                    splits = [random.randint(min_idx, max_idx-interval), ]
                    split_2 = random.randint(min(splits[0]+interval, max_idx-1), max_idx)
                    splits += [split_2]
            # Define the indices where the splits occur
            split_indices = [0] + splits + [doc_len]
            if len(split_indices) > 2:
                for _i in range(len(split_indices)-1):
                    assert split_indices[_i]+5 < split_indices[_i+1]
            # Split the input_ids and recompute eos indices
            for chunk_order in range(len(split_indices) - 1):
                chunk_input_ids = input_ids[split_indices[chunk_order]:split_indices[chunk_order + 1]]
                all_chunks.append({
                    'doc_id': doc_id,
                    'chunk_order': chunk_order,
                    'chunk_input_ids': chunk_input_ids,
                })

        # Shuffle the chunks while maintaining the order within each document
        # Assign a random base value to each document
        doc_random_bases = {doc_id: [random.random(), math.exp(random.random())] for doc_id in range(len(eos_indices)-1)}

        # Assign a sorting key to each chunk
        for chunk in all_chunks:
            doc_id = chunk['doc_id']
            chunk_order = chunk['chunk_order']
            # The key is a combination of a random base for the document and the chunk order
            # The chunk order is added as a small increment to preserve order within the document
            chunk['key'] = doc_random_bases[doc_id][0] + chunk_order * doc_random_bases[doc_id][1]

        # Sort the chunks based on the key
        all_chunks.sort(key=lambda x: x['key'])

        # Reconstruct the final input_ids and eos_indices
        reconstruct_input_ids = []
        reconstruct_eos_indices = []
        reconstruct_doc_ids = []
        for chunk in all_chunks:
            reconstruct_input_ids.extend(chunk['chunk_input_ids']) # TODO: may need to add bos for each chunk
            reconstruct_eos_indices.append(len(chunk['chunk_input_ids']) + (reconstruct_eos_indices[-1] if len(reconstruct_eos_indices) > 0 else 0))
            reconstruct_doc_ids.append(chunk['doc_id'])

        final_input_ids.append(reconstruct_input_ids)
        final_eos_indices.append(reconstruct_eos_indices)
        final_doc_ids.append(reconstruct_doc_ids)
        
    # Update model_inputs
    model_inputs['input_ids'] = final_input_ids
    model_inputs['source_eos'] = final_eos_indices
    model_inputs['source_doc_ids'] = final_doc_ids

    return model_inputs



def flex_causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def generate_doc_mask_mod(mask_mod, document_id, use_anchor_attn=True):
    def anchor_attn_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        anchor_token = document_id[kv_idx] == 0
        inner_mask = mask_mod(b, h, q_idx, kv_idx)
        return (same_doc | anchor_token) & inner_mask
    def intra_doc_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        inner_mask = mask_mod(b, h, q_idx, kv_idx)
        return same_doc & inner_mask
    
    if use_anchor_attn:
        return anchor_attn_mod
    else:
        return intra_doc_mod
    

def main(args):
    set_seed(args.seed)
    model_type = model_shortname(args.model_name_or_path)
    print("Model Type:", model_type)
    assert args.batch_size == 1, "Only support batch size 1, otherwise the loader should be modified"
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb:
        import wandb
        wandb.login()

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    dir_steps_list = get_checkpoint_list(checkpoint_dir)


    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout],
    )
    accelerator.init_trackers(project_name=args.wandb, init_kwargs={"wandb":{"name":args.output_dir.split("/")[-1]}})
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")
    setup_debugpy(accelerator)

    print(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    if args.attn_engine == "flash":
        config._attn_implementation = "flash_attention_2"
    else:
        config._attn_implementation = "flex_attention"

    scaled_max_position_embeddings=args.seq_length
    if scaled_max_position_embeddings/args.model_max_position_embeddings > args.rope_scaling_factor:
        accelerator.print("Warning: scaled_max_position_embeddings/args.model_max_position_embeddings > args.rope_scaling_factor.")
    if args.rope_scaling_type is not None:
        config.rope_scaling={"type": args.rope_scaling_type, "factor": args.rope_scaling_factor}
    config.rope_theta = args.rope_theta
    config.torch_dtype=torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=accelerator.device, 
        torch_dtype=torch.bfloat16,
        config=config,
    )
    # NOTE: by default, AutoTokenizer set use_fast=True, but LlamaTokenizer is not. 
    # By setting use_fast=True for LlamaTokenizer, we can keep the same tokenizer as AutoTokenizer (might be used in downstreaming benchmark)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
    )
    if tokenizer.pad_token is None: # this is deprecated, only used for llama2 (because we alreayd trained llama2 with its)
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    anchor_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    

    processed_data_save_folder = current_file_dir+"/data/processed_data/{}_{}_{}_{}".format(args.dataset.split('/')[-1]\
                                                            .replace('.jsonl', '').replace('.json', '').replace('-', '_'), args.seq_length, model_type, args.attn_engine)
    print("Trying to load from: ", processed_data_save_folder)
    if not os.path.exists(processed_data_save_folder):
        accelerator.wait_for_everyone()
        try:
            train_dataset = load_dataset("json", data_files=args.dataset) if 'jsonl' in args.dataset else load_dataset(args.dataset)
        except:
            train_dataset = load_from_disk("json", data_files=args.dataset) if 'jsonl' in args.dataset else load_from_disk(args.dataset)
        if isinstance(train_dataset, DatasetDict):
            train_dataset = train_dataset["train"]
        to_remove = [col for col in train_dataset.column_names if (col != "input_ids" and col != "source" and col != "text")]
        train_dataset = train_dataset.remove_columns(to_remove)
        train_dataset = train_dataset.map(
                    preprocess_raw_data,
                    batched=True,
                    batch_size=200,
                    num_proc=30,
                    # num_proc=1,
                    remove_columns=train_dataset.column_names,
                    load_from_cache_file=False,
                    desc="processing source info on raw dataset",
                    fn_kwargs={
                            "data_format": args.data_format, # "raw_json" and "tokenized"
                            "tokenizer": tokenizer,
                            "scaled_max_position_embeddings":scaled_max_position_embeddings+1, 
                            "anchor_token": [anchor_token],
                            "attn_engine": args.attn_engine,
                            }
                )
        
        train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) == scaled_max_position_embeddings+1, num_proc=30, load_from_cache_file=False)
        train_dataset = train_dataset.add_column("labels", [[PLACEHOLDER]] * len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=42)
        if len(train_dataset) > 64000:
            print(f"To save disk space, we save 64000 (from {len(train_dataset)}) examples. If you are rich (in disk space), you can remove this line.")
            train_dataset = train_dataset.select(range(64000))
        train_dataset.save_to_disk(processed_data_save_folder)
        accelerator.wait_for_everyone() # wait for the main process to finish the data processing
    else:
        train_dataset = load_from_disk(processed_data_save_folder)
    print("Dataset Size:", len(train_dataset))

    apply_seq_parallel_monkey_patch(args.parallel_mode, model_type, args.attn_engine)
    train_loader = DataLoader(
        train_dataset.shuffle(seed=42),
        collate_fn=DataCollatorForTokenClassification(tokenizer=tokenizer, padding=False,),
        shuffle=True,
        batch_size=args.batch_size,
    )

    if args.learning_rate != 2e-5:
        accelerator.print(f"Warning: You also need to modify accelerate_configs/zero3_offload.json to change the learning rate")
        assert args.learning_rate == 2e-5
    optim = DummyOptim(model.parameters(), lr=args.learning_rate)
    scheduler = DummyScheduler(
        optim,
        num_training_steps=args.max_train_steps,
        total_num_steps=args.max_train_steps,
    )
    model, optim, scheduler = accelerator.prepare(model, optim, scheduler)
    train_loader = prepare_dataloader(args.parallel_mode, train_loader, accelerator)
    model.gradient_checkpointing_enable()

    accelerator.register_for_checkpointing(scheduler)

    # check and load the last checkpoint
    last_checkpoint_steps, last_loader_steps  = 0, 0
    if dir_steps_list[-1][0] != '':
        last_checkpoint_path, last_checkpoint_steps, last_loader_steps = dir_steps_list[-1]
        last_checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint_path)
        assert os.path.exists(last_checkpoint_path), "The last checkpoint path does not exist"
        print("Loading the last checkpoint from:", last_checkpoint_path)
        print("Last checkpoint steps:", last_checkpoint_steps)
        accelerator.load_state(last_checkpoint_path)
        set_seed(args.seed) # NOTE: reset the seed after loading the checkpoint is non-trivial

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = last_checkpoint_steps
    data_loader_steps = 0
    model.train()
    loss_func = CrossEntropyLoss(inplace_backward=True)


    import time
    start_time = time.time()
    while True:
        if completed_steps >= args.max_train_steps:
                break
        accelerator.wait_for_everyone() # wait for everyone before starting the next epoch
        for _step, batch in enumerate(train_loader):
            if data_loader_steps < last_loader_steps: # skip the first several data
                data_loader_steps += 1
                accelerator.wait_for_everyone()
                continue
            else:
                data_loader_steps += 1
            
            # the real data
            bos_token_index = batch['source_eos'][0].cpu()    
            input_ids = batch["input_ids"]
            source_doc_ids = batch['source_doc_ids']
            target_ids = input_ids
            input_ids = input_ids[..., : args.seq_length + 1][..., :-1]
            target_ids = target_ids[..., : args.seq_length + 1][..., 1:]

            # The position_ids is used for flashattn masking mechanism (different from rotary_position_ids)
            position_ids = (torch.arange(input_ids.shape[-1]).unsqueeze(0).expand(input_ids.shape[0], -1))
            doc_ids = torch.zeros_like(position_ids)
            
            if len(bos_token_index) > 1:
                for _i in range(len(bos_token_index) - 1):
                    position_ids[0, bos_token_index[_i]:bos_token_index[_i+1]] -= bos_token_index[_i]
                    doc_ids[0, bos_token_index[_i]:bos_token_index[_i+1]] = source_doc_ids[0][_i]

            if args.attn_engine == "flex":
                rotary_position_ids = (torch.arange(input_ids.shape[-1]).unsqueeze(0).expand(input_ids.shape[0], -1))
            else:
                rotary_position_ids = (torch.arange(input_ids.shape[-1]).unsqueeze(0).expand(input_ids.shape[0], -1))
                if len(bos_token_index) > 1:
                    for _i in range(len(bos_token_index) - 1):
                        if bos_token_index[_i] < len(rotary_position_ids[0]):
                            rotary_position_ids[0, bos_token_index[_i]] = 0

            get_accelerator().empty_cache()
            torch.cuda.empty_cache()  # Also ensure CUDA cache is cleared
            # shard the input_ids according to the world size and rank according to zig zag attention
            prepared = prepare_seq_parallel_inputs(
                args.parallel_mode,
                input_ids=input_ids,
                position_ids=rotary_position_ids,
                target_ids=target_ids,
                doc_ids=doc_ids,
                global_position_ids=position_ids,
                rank=accelerator.process_index,
                world_size=accelerator.num_processes,
                device=accelerator.device,
            )
            local_input_ids = prepared["local_input_ids"]
            local_position_ids = prepared["local_position_ids"]
            global_doc_ids = prepared["global_doc_ids"]
            local_target_ids = prepared["local_target_ids"]
            global_position_ids = prepared["global_position_ids"]
            flex_mask = None
            if args.attn_engine == "flex":
                doc_causal_mask = generate_doc_mask_mod(flex_causal, global_doc_ids[0])
                # NOTE:
                # We are using version 2.5.0.dev20240912+cu121 for our FLEX experiment in the paper
                # instead of version 2.6.0.dev20241112+cu121.
                #
                # If you are using the stable versions 2.5.0 or 2.5.1, you need to call `torch._dynamo.reset()`.
                # However, this can cause a slight slowdown.
                #
                # To avoid this issue, we recommend using one of the following versions:
                #   - 2.5.0.dev20240912+cu121
                #   - 2.6.0.dev20241112+cu121
                # torch._dynamo.reset()
                flex_mask = create_block_mask(doc_causal_mask, B=None, H=None, Q_LEN=global_position_ids.shape[-1], KV_LEN=global_position_ids.shape[-1], _compile=True)

            loss_log = None
            with accelerator.accumulate(model):
                logits = model(
                    local_input_ids,
                    position_ids=(local_position_ids, global_position_ids, global_doc_ids, flex_mask),
                ).logits
                loss = loss_func(
                    logits.reshape(-1, logits.shape[-1]), local_target_ids.reshape(-1)
                    )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0: # NOTE: I think a better way to use grad clip is through deepspeed
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    # pay attention here. When any seq parallel algo is turned on. This technically only log the very first chunk's loss
                    # and what is the first chunk really depends on how do you shard the sequence
                    # for zig zag attention, the first chunk contains the left most and rightmost tokens
                    # so you cannot compare the (logged) loss of dist attention and zigzag ring attention.
                    # loss_log = {"loss": loss.item(), "ppl": math.exp(loss.item())}

                    # we now try gathered loss to verify if ring attention and dist flash attention produce the same loss
                    # this may slow down the training
                    gathered_loss = accelerator.reduce(loss.clone().detach(), "mean")
                    loss_log = {
                        "loss": gathered_loss.item(),
                        "ppl": math.exp(gathered_loss.item()),
                    }
                    accelerator.log(loss_log, step=completed_steps)

                optim.step()
                scheduler.step()
                optim.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                if loss_log is not None:
                    progress_bar.set_postfix(loss_log)
                completed_steps += 1

                if args.save_per_hours > 0 and (
                    (completed_steps >= args.max_train_steps) or (time.time() - start_time > 60*60*args.save_per_hours) ):
                    start_time = time.time()
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        print('Cleaning checkpoint folder')
                        clean_checkpoint_folder(checkpoint_dir, max_keep=0)
                    _save_time = time.time()
                    print('Saving accelerator checkpoint at step:', completed_steps)
                    accelerator.save_state(os.path.join(checkpoint_dir, f"checkpoint_{completed_steps}_{data_loader_steps}"), safe_serialization=True)
                    print('Finish saving accelerator checkpoint at step:', completed_steps)
                    print('Time used for saving:', time.time() - _save_time)



            if any(completed_steps + offset >= args.min_steps_for_save and (completed_steps + offset) % args.saving_interval == 0 for offset in [10, 20, 30, 40]) and completed_steps < args.max_train_steps:
                accelerator.wait_for_everyone()
                state_dict = accelerator.get_state_dict(model)
                unwrapped_model = accelerator.unwrap_model(model)
                if accelerator.is_main_process:
                    print('in Main Process')
                    # save model with state
                    unwrapped_model.save_pretrained(
                            f"{args.output_dir}/step-{completed_steps}",
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            state_dict=state_dict,
                        )
                    # save tokenizer
                    if tokenizer is not None:
                        tokenizer.save_pretrained(f"{args.output_dir}/step-{completed_steps}")
                    # Good practice: save your training arguments together with the trained model
                    try:
                        torch.save(args, os.path.join(f"{args.output_dir}/step-{completed_steps}", TRAINING_ARGS_NAME))
                    except:
                        print('Error saving training args.')
                    print('finish saving model')
                accelerator.wait_for_everyone()
            
            if completed_steps >= args.max_train_steps:
                break

    accelerator.print(f"Training Finished")
    accelerator.end_training()

    if not os.path.exists(f"{args.output_dir}/step-{completed_steps}"):
        accelerator.print(f"Saving model to {args.output_dir}/step-{completed_steps}")
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            print('in Main Process')
            accelerator.unwrap_model(model).save_pretrained(
                f"{args.output_dir}/step-{completed_steps}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state_dict,
            )
        if tokenizer is not None:
            tokenizer.save_pretrained(f"{args.output_dir}/step-{completed_steps}")
        accelerator.print("Finally Saving at step-{} Finished.".format(completed_steps))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--output-dir", type=str, required=True)

    args.add_argument("--dataset", type=str, required=True)
    args.add_argument("--data-format", type=str, choices=["raw_json", "tokenized"], required=True,
                      help='tokenized: the input_ids are already tokenized using FranxYao/Long-Context-Data-Engineering, \
                      raw_json: the input_ids are raw text; format like {"text": "..."}.')
    args.add_argument("--seq-length", type=int, default=16384)
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--max-train-steps", type=int, default=4000)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--save_per_hours", type=int, default=3)
    args.add_argument("--saving_interval", type=int, default=500)

    args.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    args.add_argument("--model_max_position_embeddings", type=int, default=4096)

    args.add_argument("--rope-theta", type=float, default=100000)
    args.add_argument("--rope_scaling_type", type=str, default=None)
    args.add_argument("--rope_scaling_factor", type=float, default=1.0)
    
    args.add_argument("--max_grad_norm", type=float, default=0)
    args.add_argument("--min_steps_for_save", type=int, default=500)

    args.add_argument("--parallel_mode", type=str, choices=["ulysses_attn", "data_parallel"], required=True)
    args.add_argument("--attn_engine", type=str, choices=["flash", "flex"], required=True)

    main(args.parse_args())
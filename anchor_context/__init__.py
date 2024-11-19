from .ulysses_attn.prepare_inputs import prepare_ulysses_attn_inputs  
from .ulysses_attn.monkey_patch import apply_ulysses_attn_monkey_patch_llama 
from .ulysses_attn.monkey_patch import apply_ulysses_attn_monkey_patch_qwen
from .ulysses_attn.monkey_patch import apply_ulysses_attn_monkey_patch_mistral


def prepare_seq_parallel_inputs(
    seq_algo, input_ids, position_ids, target_ids, doc_ids, global_position_ids, rank, world_size, device
):
    if seq_algo == "ulysses_attn":
        return prepare_ulysses_attn_inputs(
            input_ids, position_ids, target_ids, doc_ids, global_position_ids, rank, world_size, device
        )
    elif seq_algo == "data_parallel":
        return {
            "local_input_ids": input_ids.to(device),
            "local_position_ids": position_ids.to(device),
            "local_target_ids": target_ids.to(device),
            "global_doc_ids": doc_ids.to(device),
            "global_position_ids": global_position_ids.to(device),
        }
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo}")
    

def apply_seq_parallel_monkey_patch(
    seq_algo, model, attn_engine
):
    assert seq_algo in ["ulysses_attn", "data_parallel"], f"Invalid seq_algo: {seq_algo}"
    if seq_algo == "data_parallel":
        return
    elif seq_algo == "ulysses_attn" and "llama" in model:
        apply_ulysses_attn_monkey_patch_llama(attn_engine)
    elif seq_algo == "ulysses_attn" and "qwen" in model:
        apply_ulysses_attn_monkey_patch_qwen(attn_engine)
    elif seq_algo == "ulysses_attn" and "mistral" in model:
        apply_ulysses_attn_monkey_patch_mistral(attn_engine)
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo} or model: {model}")
        

def prepare_dataloader(seq_algo, dataloader, acclerator):
    if seq_algo == "data_parallel":
        return acclerator.prepare(dataloader)
    else:
        return dataloader
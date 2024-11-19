import transformers
from typing import List, Optional, Tuple, Union
import warnings
import torch
import torch.utils.checkpoint
from yunchang.ulysses import UlyssesAttention
ulysses_attn = UlyssesAttention()
from yunchang.ulysses import VarLenUlyssesAttention, UlyssesFlexAttention
varlen_ulysses_attn = VarLenUlyssesAttention()
ulysses_flex_attn = UlyssesFlexAttention()

import os
import inspect
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal
if is_flash_attn_2_available():
    from flash_attn.bert_padding import pad_input  # noqa
    from flash_attn import flash_attn_func, flash_attn_varlen_func
_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
from transformers.modeling_flash_attention_utils import (
    prepare_fa2_from_position_ids,
    _upad_input,
)

def new_flex_attn_forward(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    flex_mask = None,
):
    attn_output = ulysses_flex_attn(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, flex_mask=flex_mask
        )
    return attn_output



def new_flash_attn_forward(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    global_position_ids: Optional[torch.LongTensor] = None,
):
    
    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        causal = self.is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    # Contains at least one padding token in the sequence
    assert attention_mask is None
    assert causal is True
    assert use_sliding_windows is False


    if is_flash_attn_greater_or_equal("2.4.1"):
        if deterministic is None:
            deterministic = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        flash_kwargs["deterministic"] = deterministic

    # if softcap is not None:
    #     flash_kwargs["softcap"] = softcap
    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens=cu_seq_lens,
            max_seqlen=max_seqlen_in_batch_q,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    # If position_ids is provided and check all examples do not contain only 1 sequence, If tensor in increasing
    # then we probably have one sequence, otherwise it is packed. Additionally check we are in pre-fill/training stage.
    # Use `flash_attn_varlen_func` to prevent cross-example attention and also allow padding free approach
    elif global_position_ids is not None and not (torch.diff(global_position_ids, dim=-1) >= 0).all() and query_length != 1:
        batch_size = query_states.size(0)
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = prepare_fa2_from_position_ids(
            query_states, key_states, value_states, global_position_ids
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
        attn_output = varlen_ulysses_attn(
            query_states.unsqueeze(0),
            key_states.unsqueeze(0),
            value_states.unsqueeze(0),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))

    else:
        attn_output = ulysses_attn( # zigzag_ring_flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal, **flash_kwargs
        )

    return attn_output




def new_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    global_position_ids=None,
    flex_mask=None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    # assert isinstance(
    #     self.self_attn, transformers.models.llama.modeling_llama.LlamaFlashAttention2
    # ) or isinstance(
    #     self.self_attn,
    #     transformers.models.mistral.modeling_mistral.MistralFlashAttention2,
    # ), "Please toggle on the Flash Attention 2 implementation when using zigzag ring attention monkey patch."

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        global_position_ids=global_position_ids,
        flex_mask=flex_mask,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs




def new_decoder_forward_with_position_embeddings(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    global_position_ids=None,
    flex_mask=None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        global_position_ids=global_position_ids,
        flex_mask=flex_mask,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs



def apply_ulysses_attn_monkey_patch_llama(attn_engine):
    if attn_engine == 'flash':
        transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward = (
            new_flash_attn_forward
        )
    else:
        transformers.models.llama.modeling_llama.LlamaFlexAttention._flex_attention_forward = (
            new_flex_attn_forward
        )
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = (
        new_decoder_forward
    )



def apply_ulysses_attn_monkey_patch_qwen(attn_engine):
    if attn_engine == 'flash':
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2._flash_attention_forward = (
            new_flash_attn_forward
        )
    else:
        transformers.models.qwen2.modeling_qwen2.Qwen2FlexAttention._flex_attention_forward = (
            new_flex_attn_forward
        )
    transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer.forward = (
        new_decoder_forward_with_position_embeddings
    )



def apply_ulysses_attn_monkey_patch_mistral(attn_engine):
    if attn_engine == 'flash':
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2._flash_attention_forward = (
            new_flash_attn_forward
        )
    else:
        transformers.models.mistral.modeling_mistral.MistralFlexAttention._flex_attention_forward = (
            new_flex_attn_forward
        )
    transformers.models.mistral.modeling_mistral.MistralDecoderLayer.forward = (
        new_decoder_forward
    )


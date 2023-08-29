# Copyright (c) OpenMMLab. All rights reserved.
import logging
from types import MethodType
from enum import Enum
import torch
import torch.nn as nn

from .attn_varlen import MemoryEfficientAttentionVarlen

from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask, BlockDiagonalMask)

logger = logging.getLogger(__name__)

TABLE = {
    'LlamaAttention': (MemoryEfficientAttentionVarlen, (), {}),
}


def replace_layer(model: nn.Module):
    for name, orig in list(model.named_children()):
        CLS, args, kwargs = TABLE.get(orig.__class__.__name__, (None, [], {}))

        if CLS is None:
            replace_layer(orig)
        else:
            new = CLS(orig, *args, **kwargs)
            setattr(model, name, new)

            logger.debug(f'Replace {name} with {CLS.__name__}.')

    if hasattr(model, '_prepare_decoder_attention_mask'):

        def _prepare_decoder_attention_mask(self, attention_mask, *args,
                                            **kwargs):
            return attention_mask

        #     # attention_mask.shape == [bs, seq_len]
        #     seq_lens_q = attention_mask.sum(dim=-1)  # [bs]
        #     logger.debug(f'seq_lens_q = {seq_lens_q}')
        #     seq_lens_q_list = seq_lens_q.to(torch.int32).tolist()
        #     attn_bias = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens( # noqa
        #         q_seqlen=seq_lens_q_list,
        #         kv_padding=attention_mask.size(1),
        #         kv_seqlen=seq_lens_q_list)
        #     # # prepare inputs
        #     # input_ids = torch.tensor(sum(input_ids, [])).unsqueeze(0)
        #     # # print(f"input_ids.shape = {input_ids.shape}")
        #     # # input_ids.shape == [1, pack_len]
        #     # # input_lens = torch.tensor(input_lens, device=gpu_id)
        #     # attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
        #     #     input_lens, input_lens).make_causal()
        #     # probs = decode_single(model, input_ids, input_lens, attn_bias)

        #     return attn_bias

        noop_on_attn = MethodType(_prepare_decoder_attention_mask, model)
        model._prepare_decoder_attention_mask = noop_on_attn

    return model


def replace_model(model: nn.Module):
    for name, orig in model.named_children():
        if orig.__class__.__name__.endswith('Model'):
            setattr(model, name, ModelWrapper(orig))
            logger.debug(f'Replace {name} with ModelWrapper.')
            break
        else:
            replace_model(orig)
    return model


class InputType(Enum):
    LIST_OF_LIST = 1
    TENSOR = 2
    OTHERS = 3


class ModelWrapper(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

    @staticmethod
    def _keep_attn_mask_as_is(attention_mask, *args, **kwargs):
        return attention_mask

    @staticmethod
    def _replace_attn_mask_to_xformer_attn_bias(attention_mask,
                                                *args,
                                                is_pad_removed=True,
                                                **kwargs):
        # attention_mask.shape == [bs, seq_len]
        seq_lens_q = attention_mask.sum(dim=-1)  # [bs]
        # logger.debug(f'seq_lens_q = {seq_lens_q}')
        seq_lens_q_list = seq_lens_q.to(torch.int32).tolist()

        if is_pad_removed:
            seq_lens_kv_list = seq_lens_q_list
            attn_bias = BlockDiagonalMask.from_seqlens(
                seq_lens_q_list,
                seq_lens_kv_list).make_causal_from_bottomright()
        else:
            attn_bias = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(  # noqa
                q_seqlen=seq_lens_q_list,
                kv_padding=attention_mask.size(1),
                kv_seqlen=seq_lens_q_list)

        # # prepare inputs
        # input_ids = torch.tensor(sum(input_ids, [])).unsqueeze(0)
        # # print(f"input_ids.shape = {input_ids.shape}")
        # # input_ids.shape == [1, pack_len]
        # # input_lens = torch.tensor(input_lens, device=gpu_id)
        # attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
        #     input_lens, input_lens).make_causal()
        # probs = decode_single(model, input_ids, input_lens, attn_bias)

        return attn_bias

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        ## pre-process

        past_key_values = kwargs.pop('past_key_values', None)
        position_ids = kwargs.pop('position_ids', None)
        input_type = None

        if isinstance(input_ids, list):
            input_type = 'list_of_list'

            # input and mask are both list of list of ints
            # this is currently not supported by PretrainedModels
            input_lens = [len(x) for x in input_ids]
            input_ids = torch.tensor(sum(input_ids, [])).unsqueeze(0)
            input_ids = input_ids.cuda(self.device)

            attention_mask = BlockDiagonalMask.from_seqlens(
                input_lens, input_lens).make_causal()

            if position_ids is None:
                position_ids = torch.cat([
                    torch.arange(il, device=self.device) for il in input_lens
                ])
                position_ids = position_ids.unsqueeze(0)

            self.model._prepare_decoder_attention_mask = self._keep_attn_mask_as_is

        elif isinstance(input_ids, torch.Tensor) and isinstance(
                attention_mask, torch.Tensor):
            input_type = 'tensor'
            # input and mask are both torch tensors
            # Commonly returned by tokenizer with padding
            # Here remove padding and pack input ids

            if position_ids is None:
                position_ids = attention_mask.cumsum(dim=-1) - 1
                attention_mask_bool = attention_mask.to(torch.bool)
                input_ids = input_ids[attention_mask_bool].unsqueeze(0)
                position_ids = position_ids[attention_mask_bool].unsqueeze(0)

            print("input_ids.shape in ModelWrapper = ", input_ids.shape)
            print("position_ids.shape in ModelWrapper = ", position_ids.shape)

            self.model._prepare_decoder_attention_mask = self._replace_attn_mask_to_xformer_attn_bias
        else:
            raise ValueError(
                'Input and mask must be list of list of ints or torch tensors')

        ## original forward of xxxModel
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             past_key_values=past_key_values,
                             *args,
                             **kwargs)

        ## post-process
        if input_type == 'list_of_list':
            # return as packed, with bs=1
            start = 0
            # for i, il in enumerate(input_lens):
            #     end = start + il
            #     outputs.last_hidden_state[:, start:end, :] *= attention_mask[
            #         i].unsqueeze(-1)
            #     start = end
        elif input_type == "tensor":
            # Recover shape of input_ids and pad with 0
            idx = attention_mask.reshape(-1).cumsum(dim=-1) - 1
            outputs.last_hidden_state = outputs.last_hidden_state[:, idx, :].reshape(
                attention_mask.size(0), attention_mask.size(1),
                outputs.last_hidden_state.size(2))
            outputs.last_hidden_state *= attention_mask.unsqueeze(-1)
        else:
            raise ValueError("Unexpected.")

        return outputs

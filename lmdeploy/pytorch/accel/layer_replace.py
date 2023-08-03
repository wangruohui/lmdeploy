# Copyright (c) OpenMMLab. All rights reserved.
import logging
from types import MethodType

import torch.nn as nn

from .attn_varlen import MemoryEfficientAttentionVarlen

# from xformers.ops.fmha.attn_bias import (
#     BlockDiagonalCausalWithOffsetPaddedKeysMask, BlockDiagonalMask)

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

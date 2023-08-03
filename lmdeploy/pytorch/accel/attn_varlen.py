# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import xformers.ops as xops

from .rot_emb import triton_rotate_half_


class MemoryEfficientAttentionVarlen(nn.Module):

    def __init__(self, orig):
        super().__init__()

        # currently support this only
        assert orig.__class__.__name__ == 'LlamaAttention', \
            f'{orig.__class__.__name__} not supported.'

        self.qkv_mat = nn.Parameter(
            torch.stack(
                [orig.q_proj.weight, orig.k_proj.weight, orig.v_proj.weight],
                dim=0))  # [3, out, in]
        self.o_proj = orig.o_proj

        self.hidden_size = orig.hidden_size
        self.num_heads = orig.num_heads
        self.head_dim = orig.head_dim

    def forward(
        self,
        hidden_states,
        position_ids,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
        mode='decode',
        **kwargs,
    ):
        """Forward function.

        Below we assume bs = 1

        Args:
            hidden_states: (bs, seqlen, head * dim)
            position_ids: (bs, seqlen), position ids for rotary embedding
            past_key_value: tuple of (k_cache, v_cache), each of shape
                (bs, cachelen, head, dim)
        """
        # print(f'attention_mask = {attention_mask}')
        # print(f'position_ids.shape in VarLenAttn = {position_ids.shape}')
        bs, seqlen, _ = hidden_states.shape
        # cachelen = (
        #     past_key_value[0].shape[2] if
        #     (use_cache and past_key_value is not None) else 0)

        assert position_ids.shape == (
            bs, seqlen), f'position_ids.shape = {position_ids.shape}'

        # QKV Gemm
        qkv = torch.einsum('bsh,toh->bsto', hidden_states, self.qkv_mat)
        qkv = qkv.reshape(bs, seqlen, 3, self.num_heads, self.head_dim)

        # RoPE
        qk = qkv[:, :, :2]
        triton_rotate_half_(qk, position_ids)

        # Split QKV
        q, k, v = torch.unbind(qkv, dim=2)

        # print(f'q.shape = {q.shape}')
        # print(f'k.shape = {k.shape}')
        # print(f'v.shape = {v.shape}')

        if mode == 'decode':
            output = xops.memory_efficient_attention_forward(
                q.reshape(1, -1, q.size(2), q.size(3)),
                k.reshape(1, -1, k.size(2), k.size(3)),
                v.reshape(1, -1, v.size(2), v.size(3)),
                attn_bias=attention_mask,
            )
        else:
            raise NotImplementedError

        # print(f"output.shape = {output.shape}")
        # O proj
        output = output.reshape(bs, seqlen, self.hidden_size)
        output = self.o_proj(output)

        return output, None, None

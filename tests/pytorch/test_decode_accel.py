import logging
from typing import List

import torch
import xformers.ops as xops
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from lmdeploy.pytorch.decode import Engine, decode_single
from lmdeploy.pytorch.model import accel_model, init_model

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def _test(accel=None, gpu_id=0):

    model_path = 'llama2/huggingface/llama-2-7b'

    prompt = [
        'I believe the meaning of life is to find your gift. The purpose of life is to give it away.',  # noqa: E501
        'Simply put, the theory of relativity states that '
    ] * 8

    model, tokenizer = init_model(model_path)
    model = model.eval()
    model = accel_model(model, accel=accel)

    inputs = tokenizer(prompt)

    input_ids: List[int] = inputs.input_ids
    # attention_mask = inputs.attention_mask
    # print(attention_mask)
    input_lens = [len(x) for x in input_ids]
    max_len = max(input_lens)

    if accel is None:
        # prepare inputs
        input_ids = [torch.tensor(p) for p in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True)
        input_lens = torch.tensor(input_lens, device=gpu_id)
        attention_mask = torch.arange(
            max_len, device=gpu_id)[None, :] < input_lens[:, None]
        # print(attention_mask)

        # forward to compute probs
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=False,
                            output_attentions=False,
                            use_cache=False,
                            return_dict=True)
            # fp32, [bs, seq_len, vocab_size]
            logits = outputs.logits
            # inplace, probs is logits
            torch.softmax(logits, dim=-1, out=logits)

        # Post Processing
        # Shift to fetch probabilities
        shift_labels = input_ids[..., 1:].contiguous()
        shift_probs = logits[..., :-1, :].contiguous()
        probs = torch.gather(shift_probs, -1, shift_labels.unsqueeze(-1))

        probs = probs.squeeze(-1)

        if attention_mask is not None:
            probs *= attention_mask[..., 1:]

        probs = probs.cpu()

        return probs
    elif accel == 'replace_layer':
        # prepare inputs
        input_ids = torch.tensor(sum(input_ids, [])).unsqueeze(0)
        print(f'input_ids.shape = {input_ids.shape}')
        # input_ids.shape == [1, pack_len]
        # input_lens = torch.tensor(input_lens, device=gpu_id)
        attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
            input_lens, input_lens).make_causal()

        # forward to compute probs
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attn_bias,
                            output_hidden_states=False,
                            output_attentions=False,
                            use_cache=False,
                            return_dict=True)
            # fp32, [bs, seq_len, vocab_size]
            logits = outputs.logits
            # inplace, probs is logits
            torch.softmax(logits, dim=-1, out=logits)
        assert logits.grad_fn is None

        # Post Processing
        # Shift to fetch probabilities
        shift_labels = input_ids[..., 1:].contiguous()
        shift_probs = logits[..., :-1, :].contiguous()
        probs = torch.gather(shift_probs, -1, shift_labels.unsqueeze(-1))

        probs = probs.squeeze(0, 2)
        probs = probs.cpu()

        print(f'probs.shape = {probs.shape}')

        start = 0
        probs_list = []
        for b in input_lens:
            end = start + b
            probs_list.append(probs[start:end - 1])
            start = end

        probs = pad_sequence(probs_list, batch_first=True)
        return probs
    else:
        raise NotImplementedError


def test():
    gpu_id = 0
    torch.set_default_device(gpu_id)
    torch.set_printoptions(linewidth=108, edgeitems=5)
    res_no_accel = _test(accel=None, gpu_id=gpu_id)
    with open('res_no_accel.txt', 'w') as f:
        print(res_no_accel, file=f)
    res_replace_layer = _test(accel='replace_layer', gpu_id=gpu_id)
    with open('res_replace_layer.txt', 'w') as f:
        print(res_replace_layer, file=f)


if __name__ == '__main__':
    test()

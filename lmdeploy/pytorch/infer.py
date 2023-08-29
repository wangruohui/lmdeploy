# Copyright (c) OpenMMLab. All rights reserved.
"""Inference model to compute probabilities."""

import logging
import json
from typing import List, Optional
import time
from tqdm import tqdm
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoTokenizer, PreTrainedTokenizerFast

from .model import accel_model, init_model

logger = logging.getLogger(__name__)

torch.set_printoptions(precision=3,
                       sci_mode=False,
                       linewidth=200,
                       profile='full')
logging.basicConfig(level=logging.DEBUG,
                    format=('%(filename)s: '
                            '%(levelname)s: '
                            '%(funcName)s(): '
                            '%(lineno)d:\t'
                            '%(message)s'),
                    filename='infer.log',
                    filemode='w')


def infer(model: PreTrainedModel,
          tokenizer: PreTrainedTokenizerBase,
          prompts: List[str],
          bs=64):
    """Inference model to compute probabilities.

    Args:
        model (PreTrainedModel): Pretrained model.
        tokenizer (PreTrainedTokenizerBase): Pretrained tokenizer.
        prompts (List[str]): List of prompts.

    Returns:
        List[torch.Tensor]: List of all probabilities, one element per prompt.
    """

    # Tokenize all
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    all_probs = []

    for i in tqdm(range(-(-len(prompts) // bs))):
        logger.info(i)
        logger.info(i * bs)
        logger.info(min((i + 1) * bs, len(prompts)))
        sub_p = prompts[i * bs:min((i + 1) * bs, len(prompts))]
        inputs = tokenizer(sub_p, return_tensors="pt", padding=True)
        if inputs['input_ids'].size(-1) > 1024:
            continue
        logger.info("inputs")
        logger.info(inputs['input_ids'].shape)
        logger.info(inputs['attention_mask'].shape)

        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        with torch.no_grad():
            output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           output_hidden_states=False,
                           output_attentions=False,
                           use_cache=False)
        logits = output.logits
        logger.info("logits")
        logger.info(logits.shape)
        logger.debug(logits)

        probs = torch.softmax(logits, dim=-1)
        shift_labels = input_ids[..., 1:].contiguous()
        shift_probs = probs[..., :-1, :].contiguous()
        probs = torch.gather(shift_probs, -1, shift_labels.unsqueeze(-1))
        probs = probs.squeeze(-1)

        logger.debug(probs)

        probs *= attention_mask[..., 1:]
        logger.debug(probs)
        # float32, bs*max_len*vocab_size, e.g. 64*1k*32K=2G
        # probs = probs.half()  #

        # torch.cuda.synchronize()
        probs = probs.cpu()

        for j, p in enumerate(probs):
            # logger.info("p")
            len_ = inputs['attention_mask'][j].sum().long().item()
            p = p[:len_]
            p = p.numpy()
            # logger.info(p.shape)
            # logger.debug(p)
            all_probs.append(p)

    return all_probs


def decode(model_path,
           prompts,
           tokenizer_path: str = None,
           accel: Optional[str] = None):
    model, tokenizer = init_model(model_path, tokenizer_path)
    model = model.eval()
    model = model.cuda()
    # model = accel_model(model, accel=None, max_alloc=1024)

    with torch.no_grad():
        res = infer(model, tokenizer, prompts)

    return res


def benchmark(path='llama2/huggingface/llama-2-7b',
              share_gpt="ShareGPT_V3_unfiltered_cleaned_split.json"):

    start = time.monotonic()
    content = json.load(open(share_gpt, 'r'))

    texts = []
    for c in content:
        for cc in c['conversations']:
            texts.append(cc['value'])

    print(f"Parse json in {time.monotonic() - start} seconds.")

    # tokenizer = AutoTokenizer.from_pretrained("llama2/huggingface/llama-2-7b")

    # start = time.monotonic()
    # inputs = tokenizer(texts, return_tensors="pt", padding=False)
    # print(f"Tokenize all in {time.monotonic() - start} seconds.")

    start = time.monotonic()
    res = decode(path, texts[:2048], accel=None)
    decode_time = time.monotonic() - start
    print(f"Decode in {decode_time} seconds.")

    # total_tokens = sum([r.shape[0] for r in res])
    print(f"Decode in {decode_time} seconds.")
    for r in res:
        logger.debug(r.shape)


if __name__ == '__main__':
    benchmark()
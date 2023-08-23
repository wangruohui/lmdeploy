import numpy as np
import torch
from transformers import AutoTokenizer

from lmdeploy.pytorch.decode import Engine, decode_single
from lmdeploy.pytorch.model import init_model


def _test_decode_dist(model_path, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    inputs = tokenizer(prompt)
    input_ids = inputs.input_ids

    engine = Engine(model_path, tokenizer=tokenizer)
    probs = engine.decode(input_ids, sort=False, max_bs=4, pad=True)

    return probs


def _test_decode_single(model_path, prompt):
    model, tokenizer = init_model(model_path)
    model = model.eval()

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()

    probs: torch.Tensor = decode_single(model, input_ids, attention_mask)

    return probs.numpy()


def test_compare():
    gpu_id = 0

    torch.set_default_device(gpu_id)
    torch.set_printoptions(linewidth=200, edgeitems=5)
    np.set_printoptions(linewidth=200, edgeitems=5)

    model_path = 'llama2/huggingface/llama-2-7b'

    prompts = [
        'I believe the meaning of life is to find your gift. The purpose of life is to give it away.',  # noqa: E501
        'Simply put, the theory of relativity states that ',
        'Building a website can be done in 10 simple steps:'
    ]

    p_dist = _test_decode_dist(model_path, prompts)
    p_single = _test_decode_single(model_path, prompts)

    # print(p_single[0])
    # print(p_dist[0])

    # print(p_single[1])
    # print(p_dist[1])

    rtol = 2.0e-2
    atol = 2.0e-2
    failed = (abs(p_dist - p_single) > atol + rtol * abs(p_single))
    idx = failed.nonzero()
    print(idx)
    print(p_dist[idx])
    print(p_single[idx])

    assert np.allclose(p_dist, p_single, rtol=rtol, atol=atol)

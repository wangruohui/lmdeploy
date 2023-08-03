import json
import pickle
import time
from pathlib import Path

import fire
from transformers import AutoTokenizer

from lmdeploy.pytorch.decode import Engine


def benchmark(model_path,
              share_gpt_path,
              downsample=100,
              accel=None,
              save_to='decode_result.txt'):
    """Benchmark using ShareGPT data.

    Please download `ShareGPT_V3_unfiltered_cleaned_split.json` as data for
    this benchmark.
    """

    start = time.monotonic()
    content = json.load(open(share_gpt_path, 'r'))

    texts = []
    for c in content:
        for cc in c['conversations']:
            t = cc['value']
            # remove empty texts
            if t:
                texts.append(t)

    print(f'Parse json in {time.monotonic() - start} seconds.')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    texts = texts[::downsample]
    input_ids = tokenizer(texts, padding=False).input_ids

    print(F'Number of prompts: {len(input_ids)}')
    print(F'Maximum length: {max(map(len, input_ids))}')
    print(F'Minimum length: {min(map(len, input_ids))}')
    print(F'Total length: {sum(map(len, input_ids))}')

    if accel is None:
        bytes_per_token = 3e6
        max_bs = 1024
    elif accel == 'replace_layer':
        bytes_per_token = 5e5
        max_bs = 8192

    start = time.monotonic()
    # Init an engine
    engine = Engine(model_path,
                    tokenizer=tokenizer,
                    accel=accel,
                    bytes_per_token=bytes_per_token,
                    max_bs=max_bs)
    # decode prompts
    probs = engine.decode(input_ids, pad=False)
    total_tokens = sum(map(len, input_ids))

    elapsed = time.monotonic() - start
    print(f'Decoded {total_tokens} tokens in {elapsed:.1f} seconds, '
          f'{total_tokens / elapsed:.1f} tokens/s.')
    print(f'Decoded {len(probs)} prompts in {elapsed:.1f} seconds, '
          f'{len(probs) / elapsed:.1f} requests/s.')

    del engine

    if downsample >= 1000:
        # save probs to pkl
        pkl_path = Path(save_to).with_suffix('.pkl')

        with pkl_path.open('wb') as f:
            pickle.dump(probs, f)

        # save probs to txt
        txt_path = Path(save_to).with_suffix('.txt')
        with txt_path.open('w') as f:
            for i, p in zip(input_ids, probs):
                len_ = max(len(i) - 1, 0)
                fmt = '%.4e ' * len_ + '\n'
                assert len(p) >= len_ or len(p) == 0
                text = fmt % tuple(p[:len_])
                f.write(text)

        # np.savetxt(txt_path.as_posix(), probs, fmt='%.4e')


if __name__ == '__main__':
    fire.Fire(benchmark)

    # llama-2 on 1 A100:
    # data = ShareGPT, downsample = 100
    # Decoded 1579536 tokens in 175.3 seconds, 9012.821089984884 tokens/s.
    # Decoded 7022 prompts in 175.3 seconds, 40.067481648961376 requests/s.

    # llama-2 on 3 A100:
    # data = ShareGPT, downsample = 100
    # Decoded 1579536 tokens in 77.9 seconds, 20268.736076299527 tokens/s.
    # Decoded 7022 prompts in 77.9 seconds, 90.10688248180179 requests/s.

    # llama-2 on 8 A100:
    # data = ShareGPT, downsample = 100
    # Decoded 1579536 tokens in 55.2 seconds, 28630.35872677815 tokens/s.
    # Decoded 7022 prompts in 55.2 seconds, 127.27939026361929 requests/s.

    # llama-2 on 8 A100:
    # data = ShareGPT, downsample = 10
    # Decoded 15991314 tokens in 242.7 seconds, 65893.38488718234 tokens/s.
    # Decoded 70216 prompts in 242.7 seconds, 289.33018970413536 requests/s.

    # Above time all includes time for workers to load model.

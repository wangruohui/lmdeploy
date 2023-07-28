# Copyright (c) OpenMMLab. All rights reserved.
"""Chat through command line.

This submodule allow user to chat with language model through command line,
and optionally accelerate model using backends like deepspeed.

Example 1: Chat with default setting

```python
python -m lmdeploy.pytorch.chat $PATH_TO_HF_MODEL
```

Example 2: Disable sampling and chat history

```python
python -m lmdeploy.pytorch.chat \
    $PATH_TO_LLAMA_MODEL_IN_HF_FORMAT \
    --temperature 0 --max-histroy 0
```

Example 3: Accelerate with deepspeed inference

```python
python -m lmdeploy.pytorch.chat \
    $PATH_TO_LLAMA_MODEL_IN_HF_FORMAT \
    --accel deepspeed
```

Note: to use deepspeed, you need to install deepspeed,
    and if hope to accelerate InternLM, you need a customized version
    https://github.com/wangruohui/DeepSpeed/tree/support_internlm_0.10.0

Example 4: Tensor parallel the model on 2 GPUs

```python
deepspeed --module --num_gpus 2 lmdeploy.pytorch.chat \
    $PATH_TO_LLAMA_MODEL_IN_HF_FORMAT \
    --accel deepspeed \
```

This module also allow the following control commands to change
generation behaviors during chat.

- `exit`: terminate and exit chat
- `config set key=value`: change generation config `key` to `value`,
    e.g. config temperature=0 disable sampling for following chats
- `clear`: clear chat history
"""

import itertools
import logging
from typing import Optional

import fire
import torch
from transformers import GenerationConfig, PreTrainedModel

from .adapters import init_adapter
from .dist import get_local_rank, get_rank, get_world_size
from .model import accel_model, init_model
from .session import BasicSessionManager  # noqa: F401
from .session import BasicSessionManagerWithHistory
from .utils import BasicStreamer, TerminalIO, control

logger = logging.getLogger(__name__)


def set_logging(log_file, debug: bool):
    torch.set_printoptions(linewidth=120)
    level = logging.DEBUG if debug else logging.INFO
    log_file = log_file or 'chat.log'
    if r := get_rank() != 0:
        log_file = log_file + f'.{r}'
    logging.basicConfig(level=level,
                        format=('%(filename)s: '
                                '%(levelname)s: '
                                '%(funcName)s(): '
                                '%(lineno)d:\t'
                                '%(message)s'),
                        filename=log_file,
                        filemode='w')
    print(f'Worker {get_rank()} logging to {log_file}')


def main(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    accel: Optional[str] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    use_fast_tokenizer: bool = True,
    max_alloc: int = 2048,
    max_history: int = None,
    log_file: Optional[str] = None,
    debug: bool = True,
    adapter: Optional[str] = None,
):
    """Chat with model through terminal.

    Args:
        model_path (str): Path to model.
        tokenizer_path (str): Path to tokenizer.
        accel (str): Model accelerator.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Temperature for sampling.
        top_p (float): Top p for sampling.
        seed (int): Random seed.
        use_fast_tokenizer (bool): Whether to use fast tokenizer.
    """
    set_logging(log_file, debug)

    # workers should sync in sampling
    torch.manual_seed(seed)

    local_rank = get_local_rank()
    world_size = get_world_size()

    # Init model and tokenizer
    if not tokenizer_path:
        tokenizer_path = model_path

    model, tokenizer = init_model(
        model_path,
        tokenizer_path,
        use_fast_tokenizer=use_fast_tokenizer,
    )

    # Init adapter based on model and tokenizer
    adapter = init_adapter(model, tokenizer, adapter)

    # Accelerate model
    model: PreTrainedModel = accel_model(model,
                                         accel,
                                         max_alloc=max_alloc,
                                         tp_size=world_size)

    # warmup
    warmup_config = GenerationConfig(
        max_new_tokens=1,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
    )
    model.generate(torch.tensor([[6]], device=get_local_rank()), warmup_config)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
    )

    # Session manager handling history
    sm = BasicSessionManagerWithHistory(max_history=max_history or max_alloc,
                                        start_ids=adapter.start_ids,
                                        sep_ids=adapter.sep_ids)
    io = TerminalIO()
    streamer = BasicStreamer(adapter.decode, io.output)

    for r in itertools.count(1):
        # User input from IO
        logger.info(f'Round {r}')

        prompt: str = io.input()
        logger.info(f'User input: {prompt}')

        # Allow user to change config during runtime or exit
        if control(prompt, gen_config, sm):
            continue

        # Tokenize and apply model specific templates
        input_ids = adapter.encode_and_decorate(prompt)
        logger.info(f'Input ids:\n{input_ids}')

        # Prepend chat history (tensor concatenation)
        input_ids = sm.prepend_history(input_ids)
        logger.info(f'Input ids with history:\n{input_ids}')

        # Generate
        input_ids = input_ids.cuda(local_rank)
        # returned tensor including input and generated output
        output = model.generate(input_ids,
                                gen_config,
                                streamer=streamer,
                                stopping_criteria=adapter.stopping_criteria)
        logger.info(f'Output:\n{output}')

        # Save output into session manager and maybe trim some history
        sm.add_to_history(output)


def cli():
    fire.Fire(main)


if __name__ == '__main__':
    cli()

# Copyright (c) OpenMMLab. All rights reserved.
import logging
from types import MethodType

import torch.nn as nn

from .attn_varlen import MemoryEfficientAttentionVarlen

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

        noop_on_attn = MethodType(_prepare_decoder_attention_mask, model)
        model._prepare_decoder_attention_mask = noop_on_attn

    return model

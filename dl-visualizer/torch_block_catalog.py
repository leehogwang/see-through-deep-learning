#!/usr/bin/env python3
import inspect
import json
import re
from typing import Any

import torch.nn as nn


SKIP_PARAMS = {'device', 'dtype', 'factory_kwargs'}
CATEGORY_COLORS = {
    'conv': '#6366f1',
    'pooling': '#f59e0b',
    'activation': '#ec4899',
    'norm': '#14b8a6',
    'attention': '#8b5cf6',
    'recurrent': '#f97316',
    'embedding': '#06b6d4',
    'transformer': '#a855f7',
    'shape': '#64748b',
    'regularize': '#94a3b8',
    'linear': '#3b82f6',
    'padding': '#64748b',
    'upsample': '#0ea5e9',
    'distance': '#f97316',
    'loss': '#ef4444',
    'container': '#64748b',
    'misc': '#475569',
}


def camel_label(name: str) -> str:
    words = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).replace('_', ' ')
    return words.strip()


def category_for(name: str) -> str:
    lower = name.lower()
    if 'loss' in lower:
        return 'loss'
    if name in {'Sequential', 'ModuleList', 'ModuleDict', 'ParameterList', 'ParameterDict', 'Container'}:
        return 'container'
    if 'pad' in lower:
        return 'padding'
    if 'upsample' in lower:
        return 'upsample'
    if any(token in lower for token in ('distance', 'similarity', 'pairwise', 'cosine')):
        return 'distance'
    if 'conv' in lower:
        return 'conv'
    if 'pool' in lower:
        return 'pooling'
    if any(token in lower for token in (
        'relu', 'gelu', 'silu', 'mish', 'sigmoid', 'softmax', 'softmin', 'softsign',
        'swish', 'hardswish', 'hardtanh', 'tanh', 'elu', 'selu', 'celu', 'glu',
        'softplus', 'softshrink', 'hardshrink', 'prelu', 'threshold', 'identity',
    )):
        return 'activation'
    if 'norm' in lower:
        return 'norm'
    if 'attention' in lower or 'attn' in lower:
        return 'attention'
    if any(token in lower for token in ('lstm', 'gru', 'rnn')):
        return 'recurrent'
    if 'embed' in lower:
        return 'embedding'
    if 'transformer' in lower:
        return 'transformer'
    if any(token in lower for token in ('dropout', 'droppath')):
        return 'regularize'
    if any(token in lower for token in ('linear', 'bilinear', 'lazylinear')):
        return 'linear'
    if any(token in lower for token in ('flatten', 'unflatten', 'reshape', 'permute', 'pixelshuffle', 'pixelunshuffle', 'fold', 'unfold', 'channelshuffle')):
        return 'shape'
    return 'misc'


def serialize_default(value: Any) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    if value is None:
        return 'None'
    if isinstance(value, (tuple, list)):
        return ', '.join('None' if item is None else str(item) for item in value)
    if hasattr(value, 'value') and isinstance(value.value, (bool, int, float, str)):
        return value.value
    return repr(value)


def inferred_default(name: str, category: str) -> str | int | float | bool:
    lower = name.lower()

    if lower in {'bias', 'affine', 'track_running_stats', 'batch_first'}:
        return True
    if lower in {'return_indices', 'ceil_mode', 'inplace'}:
        return False
    if lower in {'p', 'dropout'}:
        return 0.5
    if lower in {'drop_prob', 'drop_path'}:
        return 0.1
    if lower in {'eps'}:
        return 1e-5
    if lower in {'momentum'}:
        return 0.1
    if lower in {'num_heads', 'nhead'}:
        return 8
    if lower in {'groups', 'num_groups'}:
        return 1
    if lower in {'kernel_size', 'patch_size'}:
        return 3
    if lower in {'stride', 'dilation'}:
        return 1
    if lower in {'padding', 'output_padding'}:
        return 0
    if lower in {'output_size'}:
        return '1,1'
    if lower in {'mode'}:
        return 'nearest'
    if lower in {'padding_mode'}:
        return 'zeros'
    if lower in {'scale_factor'}:
        return 2
    if lower in {'num_embeddings', 'vocab_size', 'n_classes', 'num_classes'}:
        return 1000
    if lower in {'embed_dim', 'embedding_dim', 'd_model', 'hidden_size', 'input_size', 'in_features', 'out_features', 'normalized_shape', 'dim'}:
        return 64
    if lower in {'in_channels', 'in_channel', 'input_channels'}:
        return 3 if category == 'conv' else 64
    if lower in {'out_channels', 'out_channel'}:
        return 64
    if lower in {'num_features', 'features'}:
        return 64
    if lower in {'cutoffs'}:
        return '100, 500'
    return 1


def max_inputs_for(name: str) -> int | None:
    lower = name.lower()
    if name == 'Bilinear':
        return 2
    if 'attention' in lower or 'attn' in lower:
        return 3
    return None


def required_forward_inputs(cls: type[nn.Module]) -> int | None:
    try:
        signature = inspect.signature(cls.forward)
    except (TypeError, ValueError):
        return None

    count = 0
    positional_seen = False
    for param in signature.parameters.values():
        if param.name == 'self':
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            return None
        if param.kind not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            continue
        positional_seen = True
        if param.default is inspect._empty:
            count += 1

    if not positional_seen:
        return None
    return max(count, 1)


def description_for(name: str, category: str) -> str:
    return f'torch.nn.{name} module ({category}) discovered from the local PyTorch install.'


def build_default_params(cls: type[nn.Module]) -> dict[str, str | int | float | bool]:
    signature = inspect.signature(cls.__init__)
    params: dict[str, str | int | float | bool] = {}
    category = category_for(cls.__name__)
    for param in signature.parameters.values():
        if param.name == 'self' or param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if param.name in SKIP_PARAMS:
            continue
        if param.default is inspect._empty:
            params[param.name] = inferred_default(param.name, category)
        else:
            params[param.name] = serialize_default(param.default)
    return params


def iter_modules() -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    for name in sorted(dir(nn)):
        if name.startswith('_'):
            continue
        obj = getattr(nn, name)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, nn.Module) or obj is nn.Module or inspect.isabstract(obj):
            continue
        forward_inputs = required_forward_inputs(obj)
        if forward_inputs is None:
            continue
        category = category_for(name)
        max_inputs = max_inputs_for(name) or forward_inputs
        blocks.append({
            'type': name,
            'label': camel_label(name),
            'category': category,
            'color': CATEGORY_COLORS.get(category, CATEGORY_COLORS['misc']),
            'description': description_for(name, category),
            'defaultParams': build_default_params(obj),
            **({'maxInputs': min(max_inputs, 8)} if max_inputs > 1 else {}),
        })
    return blocks


if __name__ == '__main__':
    print(json.dumps(iter_modules()))

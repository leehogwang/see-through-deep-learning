#!/usr/bin/env python3
import copy
import importlib
import inspect
import json
import os
import sys
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from typing import Any, Optional, get_args, get_origin

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from parse_model import FUNC_BLOCK_MAP, NN_TYPE_MAP


def to_json(data):
    print(json.dumps(data))


# ── 전처리 transform 자동 감지 ────────────────────────────────────────────────

def _discover_transforms(model: nn.Module, payload: dict[str, Any]):
    """
    모델 또는 로딩 컨텍스트에서 입력 전처리 파이프라인을 자동으로 감지한다.
    감지 순서:
    1. model.transform / model.preprocess / model.transforms 속성
    2. timm: timm.data.create_transform(model.default_cfg)
    3. torchvision: weights 객체의 transforms()
    4. 모델의 첫 Conv 레이어의 in_channels/kernel_size로부터 표준 전처리 추론
    반환: list of transform-step dicts (없으면 [])
    """
    steps = []

    # ── 1. 모델 속성에서 transform 탐색 ──────────────────────────────────────
    for attr in ('transform', 'preprocess', 'transforms', '_transform', '_preprocess'):
        t = getattr(model, attr, None)
        if t is not None:
            steps = _parse_compose(t)
            if steps:
                return steps

    # ── 2. timm default_cfg ───────────────────────────────────────────────────
    try:
        import timm
        cfg = getattr(model, 'default_cfg', None) or getattr(model, 'pretrained_cfg', None)
        if cfg is not None:
            from timm.data import create_transform
            input_size = cfg.get('input_size', (3, 224, 224))
            h, w = int(input_size[1]), int(input_size[2])
            mean = cfg.get('mean', (0.485, 0.456, 0.406))
            std  = cfg.get('std',  (0.229, 0.224, 0.225))
            crop_pct = float(cfg.get('crop_pct', 0.875))
            steps = [
                _make_step('Resize', f'({int(h / crop_pct)}, {int(w / crop_pct)})', {'size': f'({int(h / crop_pct)}, {int(w / crop_pct)})', 'interpolation': 'bilinear'}, 'pretrained_cfg'),
                _make_step('CenterCrop', f'({h}, {w})',                             {'size': f'({h}, {w})'},                                                                    'pretrained_cfg'),
                _make_step('ToTensor', '', {},                                      'pretrained_cfg'),
                _make_step('Normalize', f'mean={mean}',                             {'mean': str(mean), 'std': str(std)},                                                       'pretrained_cfg'),
            ]
            return steps
    except Exception:
        pass

    # ── 3. torchvision weights API ────────────────────────────────────────────
    try:
        import torchvision.models as tvm
        class_name = model.__class__.__name__
        for attr_name in dir(tvm):
            if not attr_name.endswith('_Weights'):
                continue
            weights_cls = getattr(tvm, attr_name)
            if not hasattr(weights_cls, 'DEFAULT'):
                continue
            if class_name.lower() not in attr_name.lower():
                continue
            try:
                t = weights_cls.DEFAULT.transforms()
                parsed = _parse_compose(t)
                if parsed:
                    return parsed
                # ImageClassification 같은 단일 nn.Module transform 처리
                parsed = _parse_image_classification_transform(t)
                if parsed:
                    return parsed
            except Exception:
                pass
    except Exception:
        pass

    # ── 4. 첫 Conv 레이어로부터 표준 전처리 추론 ──────────────────────────────
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            in_ch = int(getattr(module, 'in_channels', 3))
            # 입력 크기 후보: model.image_size 또는 기본 224
            image_size = getattr(model, 'image_size', 224)
            if isinstance(image_size, (tuple, list)):
                h, w = int(image_size[0]), int(image_size[1])
            else:
                h = w = int(image_size)
            crop_h = int(round(h / 0.875))
            crop_w = int(round(w / 0.875))
            mean = (0.485, 0.456, 0.406) if in_ch == 3 else (0.5,)
            std  = (0.229, 0.224, 0.225) if in_ch == 3 else (0.5,)
            steps = [
                _make_step('Resize', f'({crop_h}, {crop_w})', {'size': f'({crop_h}, {crop_w})', 'interpolation': 'bilinear'}, 'inferred'),
                _make_step('CenterCrop', f'({h}, {w})',        {'size': f'({h}, {w})'},                                        'inferred'),
                _make_step('ToTensor', '',                     {},                                                              'inferred'),
                _make_step('Normalize', f'mean={mean}',        {'mean': str(mean), 'std': str(std)},                            'inferred'),
            ]
            return steps
        break  # 첫 번째 Conv만 확인

    return []


def _make_step(class_name: str, label_suffix: str, params: dict, source: str) -> dict:
    label = class_name + (f' {label_suffix}' if label_suffix else '')
    return {
        'id': f'preproc_{class_name.lower()}',
        'kind': 'module',
        'label': class_name,
        'block_type': class_name,
        'attr': class_name.lower(),
        'attr_path': f'preprocess.{class_name.lower()}',
        'module_class': class_name,
        'params': params,
        'source': source,
        'output_shape': '',
    }


def _parse_image_classification_transform(t: Any) -> list[dict]:
    """
    torchvision ImageClassification / VisionTransform 같이
    resize_size, crop_size, mean, std 속성을 직접 갖는 단일 transform 객체 처리.
    """
    resize = getattr(t, 'resize_size', None)
    crop   = getattr(t, 'crop_size', None)
    mean   = getattr(t, 'mean', None)
    std    = getattr(t, 'std', None)
    interp = getattr(t, 'interpolation', None)

    if resize is None and crop is None:
        return []

    steps = []
    if resize is not None:
        r = list(resize) if hasattr(resize, '__iter__') else [resize, resize]
        steps.append(_make_step(
            'Resize', f'({r[0]}, {r[0]})',
            {'size': f'({r[0]}, {r[0]})', 'interpolation': str(interp) if interp else 'bilinear'},
            'weights-api',
        ))
    if crop is not None:
        c = list(crop) if hasattr(crop, '__iter__') else [crop, crop]
        steps.append(_make_step(
            'CenterCrop', f'({c[0]}, {c[0]})',
            {'size': f'({c[0]}, {c[0]})'},
            'weights-api',
        ))
    steps.append(_make_step('ToTensor', '', {}, 'weights-api'))
    if mean is not None and std is not None:
        m = tuple(float(v) for v in mean) if hasattr(mean, '__iter__') else (float(mean),)
        s = tuple(float(v) for v in std)  if hasattr(std, '__iter__')  else (float(std),)
        steps.append(_make_step(
            'Normalize', f'mean={m}',
            {'mean': str(m), 'std': str(s)},
            'weights-api',
        ))
    return steps


def _parse_compose(t: Any) -> list[dict]:
    """torchvision Compose / nn.Sequential 을 step 리스트로 변환한다."""
    transforms_list = None
    if hasattr(t, 'transforms'):          # torchvision.transforms.Compose
        transforms_list = t.transforms
    elif hasattr(t, 'children'):          # nn.Sequential
        transforms_list = list(t.children())
    if not transforms_list:
        return []
    steps = []
    for step in transforms_list:
        class_name = step.__class__.__name__
        params = {}
        for attr in ('size', 'mean', 'std', 'scale', 'ratio', 'p', 'interpolation', 'antialias'):
            val = getattr(step, attr, None)
            if val is not None:
                params[attr] = str(val)
        steps.append(_make_step(class_name, '', params, 'model-attr'))
    return steps


def _prepend_preprocess_ops(
    ops: list[dict],
    preprocess_steps: list[dict],
    forward_inputs: list[dict],
    h: int = 224,
    w: int = 224,
    channels: int = 3,
    include_data_sampling: bool = True,
) -> list[dict]:
    """
    forward_graph ops 앞에 DataSampling + 전처리 step들을 삽입한다.
    Input(t0) → DataSampling(synth_raw_t) → [Resize → ... →] 첫 모델 레이어
    """
    if not ops:
        return ops

    if not forward_inputs:
        return ops
    input_token = forward_inputs[0]['token']  # e.g. 't0'

    prep_prefix: list[dict] = []
    prev_token = input_token

    if include_data_sampling:
        synth_token = 'synth_raw_t'
        data_sampling_op = {
            'id': 'op_data_sampling',
            'block_type': 'DataSampling',
            'label': 'DataSampling',
            'attr_path': 'preprocess.data_sampling',
            'kind': 'module',
            'params': {
                'method': 'torch.randn',
                'shape': f'(1, {channels}, {h}, {w})',
            },
            'inputs': [{'token': input_token, 'label': 'seed', 'strategy': 'chain'}],
            'output': {'token': synth_token, 'label': 'DataSampling', 'shape': f'[1, {channels}, {h}, {w}]'},
            'output_shape': f'[1, {channels}, {h}, {w}]',
        }
        prep_prefix.append(data_sampling_op)
        prev_token = synth_token

    # 전처리 체인: Input 또는 DataSampling 출력 → Resize → ... → final_token
    prep_ops: list[dict] = []
    for i, step in enumerate(preprocess_steps):
        new_token = f'preproc_t{i}'
        op = dict(step)
        op['id'] = f'op_preproc_{i}'
        op['inputs'] = [{'token': prev_token, 'label': step['label'], 'strategy': 'chain'}]
        op['output'] = {'token': new_token, 'label': step['label'], 'shape': ''}
        op['output_shape'] = ''
        prep_ops.append(op)
        prev_token = new_token

    final_token = prev_token  # 전처리 마지막 or synth_raw_t (전처리 없을 때)

    # 모든 ops에서 input_token을 참조하는 inputs를 final_token으로 교체
    for op in ops:
        for inp in op.get('inputs', []):
            if inp['token'] == input_token:
                inp['token'] = final_token

    return prep_prefix + prep_ops + ops


def normalize_block_type(module_class: str, fallback: Optional[str] = None):
    if fallback:
        return fallback
    mapped = NN_TYPE_MAP.get(module_class)
    if mapped:
        return mapped
    lower = module_class.lower()
    return FUNC_BLOCK_MAP.get(lower)


def infer_simple_params(module: nn.Module) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
        'in_features', 'out_features', 'num_features', 'num_heads',
        'embed_dim', 'hidden_size', 'num_embeddings', 'embedding_dim',
        'p', 'dim', 'eps', 'groups', 'image_size', 'patch_size',
    ):
        if hasattr(module, name):
            value = getattr(module, name)
            if isinstance(value, (int, float, str, bool)):
                result[name] = value
            elif isinstance(value, tuple) and all(isinstance(item, (int, float, str, bool)) for item in value):
                result[name] = str(value)
    return result


def is_tensor_like(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


class _DummyClipLike(nn.Module):
    """CLIP API를 흉내내는 더미 모듈.
    encode_text / encode_image 호출 시 적절한 크기의 랜덤 텐서를 반환.
    clip_dim=1024 기본값 (OpenCLIP ViT-L 등)."""
    def __init__(self, clip_dim: int = 1024):
        super().__init__()
        self._clip_dim = clip_dim
        self._dummy_param = nn.Parameter(torch.zeros(1))  # parameters()가 비지 않도록

    def encode_text(self, text_tokens):
        b = text_tokens.shape[0] if hasattr(text_tokens, 'shape') else 1
        if not hasattr(text_tokens, 'float'):
            return torch.zeros(b, self._clip_dim, device=self._dummy_param.device)
        token_features = text_tokens.float().mean(dim=1, keepdim=True)
        return token_features.repeat(1, self._clip_dim)

    def encode_image(self, images):
        b = images.shape[0] if hasattr(images, 'shape') else 1
        if not hasattr(images, 'float'):
            return torch.zeros(b, self._clip_dim, device=self._dummy_param.device)
        image_features = images.float().mean(dim=tuple(range(1, images.dim())), keepdim=False).view(b, 1)
        return image_features.repeat(1, self._clip_dim)

    def forward(self, *args, **kwargs):
        return self.encode_image(args[0]) if args else torch.zeros(1, self._clip_dim, device=self._dummy_param.device)


def flatten_tensor_tree(value: Any, prefix: str = '') -> list[tuple[str, torch.Tensor]]:
    items: list[tuple[str, torch.Tensor]] = []
    if isinstance(value, torch.Tensor):
        items.append((prefix or 'tensor', value))
        return items
    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            child_prefix = f'{prefix}[{idx}]' if prefix else f'[{idx}]'
            items.extend(flatten_tensor_tree(child, child_prefix))
        return items
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f'{prefix}.{key}' if prefix else str(key)
            items.extend(flatten_tensor_tree(child, child_prefix))
        return items
    if is_dataclass(value):
        for field in fields(value):
            child = getattr(value, field.name)
            child_prefix = f'{prefix}.{field.name}' if prefix else field.name
            items.extend(flatten_tensor_tree(child, child_prefix))
        return items
    return items


def annotation_name(annotation: Any) -> str:
    if annotation is inspect._empty:
        return ''
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, '__name__', str(annotation))


def first_leaf_module(model: nn.Module):
    for name, module in model.named_modules():
        if not name:
            continue
        if len(list(module.children())) == 0:
            return name, module
    return '', None


def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def first_input_seed_module(model: nn.Module):
    preferred_types = (
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.Embedding, nn.Linear,
    )
    leaves: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if len(list(module.children())) == 0:
            leaves.append((name, module))
    for preferred in preferred_types:
        for name, module in leaves:
            if isinstance(module, preferred):
                return name, module
    return first_leaf_module(model)


def image_hw_from_payload(payload: dict[str, Any], model: nn.Module, forced_hw: Optional[tuple[int, int]] = None) -> tuple[int, int]:
    if forced_hw:
        return forced_hw
    sample = payload.get('sample') or {}
    width = int(sample.get('width') or 0)
    height = int(sample.get('height') or 0)
    image_size = getattr(model, 'image_size', None)
    if isinstance(image_size, int):
        return image_size, image_size
    if isinstance(image_size, (tuple, list)) and len(image_size) >= 2:
        return int(image_size[0]), int(image_size[1])
    if width > 0 and height > 0:
        return height, width
    return 224, 224


def _get_input_channels(model: nn.Module) -> int:
    _, leaf = first_input_seed_module(model)
    if isinstance(leaf, (nn.Conv2d, nn.Conv3d, nn.Conv1d)):
        return int(getattr(leaf, 'in_channels', 3))
    return 3


def synth_scalar_for_name(name: str) -> Any:
    lower = name.lower()
    if 'dropout' in lower or lower == 'p':
        return 0.0
    if 'eps' in lower:
        return 1e-5
    if 'temperature' in lower:
        return 1.0
    if 'bias' in lower or lower.startswith('use_') or lower.startswith('with_'):
        return False
    if any(token in lower for token in ('num_classes', 'classes')):
        return 10
    if any(token in lower for token in ('channels', 'in_ch', 'out_ch')):
        return 3
    if any(token in lower for token in ('heads', 'nhead', 'num_heads')):
        return 4
    if any(token in lower for token in ('depth', 'layers', 'num_layers')):
        return 2
    if any(token in lower for token in ('hidden', 'embed', 'dim', 'width')):
        return 64
    if 'patch' in lower:
        return 16
    if any(token in lower for token in ('kernel', 'window')):
        return 3
    if 'stride' in lower:
        return 1
    if 'padding' in lower:
        return 1
    if 'image_size' in lower or lower == 'size':
        return 224
    if 'vocab' in lower:
        return 1000
    return 2


def synth_value_for_param(param: inspect.Parameter, model: Optional[nn.Module] = None, payload: Optional[dict[str, Any]] = None) -> Any:
    if param.default is not inspect._empty:
        return param.default

    name = param.name
    annotation = param.annotation
    origin = get_origin(annotation)
    args = get_args(annotation)
    ann_name = annotation_name(annotation).lower()

    if param.kind == inspect.Parameter.VAR_POSITIONAL:
        return []
    if param.kind == inspect.Parameter.VAR_KEYWORD:
        return {}

    if origin is Optional or (origin is not None and type(None) in args):
        return None

    if annotation in (bool,) or ann_name == 'bool':
        return False
    if annotation in (float,) or ann_name == 'float':
        return float(synth_scalar_for_name(name))
    if annotation in (int,) or ann_name == 'int':
        return int(synth_scalar_for_name(name))
    if annotation in (str,) or ann_name == 'str':
        return ''

    if origin in (tuple, list):
        base = synth_scalar_for_name(name)
        if 'image' in name.lower() or 'size' in name.lower():
            return (224, 224)
        return (base, base)

    if 'weights' in name.lower():
        return None
    if 'device' in name.lower():
        return 'cpu'
    if 'dtype' in name.lower():
        return None
    if 'config' in name.lower() or name.lower() in ('cfg', 'args', 'opt', 'opts'):
        return None

    # 파라미터 이름이 nn.Module을 기대할 경우 → CLIP-like 더미 모듈 반환
    # clip_model, image_encoder, backbone_net 등 어떤 모델이든
    # _DummyClipLike: encode_text/encode_image + parameters() 지원
    _lower_name = name.lower()
    if any(token in _lower_name for token in ('clip', 'vision', 'language', 'text_enc', 'img_enc')):
        return _DummyClipLike()
    if any(token in _lower_name for token in ('_model', '_encoder', '_backbone', '_network', '_net', '_module', '_extractor')):
        return _DummyClipLike()
    if _lower_name in ('model', 'encoder', 'backbone', 'network', 'net', 'module', 'extractor', 'feature_extractor'):
        return _DummyClipLike()

    # 텍스트/토큰 시퀀스 배치 파라미터 → 더미 리스트 반환
    # (ingredient_texts, captions, sentences, queries, token_ids 등)
    if any(token in _lower_name for token in ('_texts', '_captions', '_sentences', '_queries', '_ingredients', 'ingredient')):
        return [['dummy text a', 'dummy text b']]
    if _lower_name in ('texts', 'captions', 'sentences', 'queries', 'ingredients'):
        return [['dummy text a', 'dummy text b']]
    if any(token in _lower_name for token in ('token_ids', 'input_ids', 'attention_mask')):
        return torch.zeros(1, 16, dtype=torch.long)

    if model is not None and payload is not None:
        lower = name.lower()
        if any(token in lower for token in ('img', 'image', 'pixel', 'input', 'x', 'sample', 'samples')):
            h, w = image_hw_from_payload(payload, model)
            channels = 3
            _, leaf = first_input_seed_module(model)
            if isinstance(leaf, (nn.Conv2d, nn.Conv3d, nn.Conv1d)):
                channels = int(getattr(leaf, 'in_channels', channels))
            return torch.randn(1, channels, h, w)

    return synth_scalar_for_name(name)


def import_module_from_source(repo_root: str, source_file: str):
    sys.path.insert(0, repo_root)
    rel_path = os.path.relpath(source_file, repo_root)
    module_name = rel_path[:-3].replace(os.sep, '.')
    if module_name.endswith('.__init__'):
        module_name = module_name[:-9]
    try:
        return importlib.import_module(module_name), {
            'strategy': 'repo-source',
            'moduleName': module_name,
        }
    except Exception as source_exc:
        sys.path = [entry for entry in sys.path if entry != repo_root]
        top_level = module_name.split('.')[0]
        for key in list(sys.modules.keys()):
            if key == top_level or key.startswith(f'{top_level}.'):
                sys.modules.pop(key, None)
        try:
            return importlib.import_module(module_name), {
                'strategy': 'installed-package-mirror',
                'moduleName': module_name,
                'sourceImportError': str(source_exc),
            }
        except Exception as installed_exc:
            raise RuntimeError(f'source import failed: {source_exc}\ninstalled package import failed: {installed_exc}')


def instantiate_from_callable(callable_obj, payload: dict[str, Any]):
    signature = inspect.signature(callable_obj)
    args = []
    kwargs = {}
    for name, param in signature.parameters.items():
        if name in ('self', 'cls'):
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        value = synth_value_for_param(param, payload=payload)
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[name] = value
    return callable_obj(*args, **kwargs), {
        'args': args,
        'kwargs': kwargs,
    }


def resolve_model_instance(module, target_name: str, preferred_factory: Optional[str], payload: dict[str, Any]):
    target = getattr(module, target_name, None)
    candidates: deque[tuple[str, str, Any]] = deque()

    if preferred_factory and hasattr(module, preferred_factory):
        candidates.append(('factory', preferred_factory, getattr(module, preferred_factory)))
    if inspect.isclass(target) and issubclass(target, nn.Module):
        candidates.append(('class', target_name, target))

    for name, value in vars(module).items():
        if name.startswith('_'):
            continue
        if preferred_factory and name == preferred_factory:
            continue
        if inspect.isclass(value) or not callable(value):
            continue
        if not (inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value)):
            continue
        if getattr(value, '__module__', None) != getattr(module, '__name__', None):
            continue
        candidates.append(('factory', name, value))

    seen = set()
    errors = []
    while candidates:
        kind, name, callable_obj = candidates.popleft()
        if name in seen:
            continue
        seen.add(name)
        try:
            instance, kwargs = instantiate_from_callable(callable_obj, payload)
            if not isinstance(instance, nn.Module):
                continue
            if inspect.isclass(target) and issubclass(target, nn.Module) and not isinstance(instance, target):
                continue
            instance.eval()
            return instance, {
                'kind': kind,
                'callable': name,
                'kwargs': kwargs,
            }
        except Exception as exc:
            errors.append(f'{kind}:{name}: {exc}')

    raise RuntimeError('constructor resolution failed\n' + '\n'.join(errors[:20]))


def load_image_tensor(sample_path: str, channels: int, hw: tuple[int, int]) -> torch.Tensor:
    try:
        from PIL import Image
        image = Image.open(sample_path)
        if channels == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        if hw[0] > 0 and hw[1] > 0:
            image = image.resize((hw[1], hw[0]))
        import numpy as np
        arr = np.array(image)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        if channels == 1 and tensor.shape[0] != 1:
            tensor = tensor[:1]
        if channels == 3 and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor.unsqueeze(0)
    except Exception:
        return torch.randn(1, channels, hw[0], hw[1])


def infer_forward_inputs(model: nn.Module, payload: dict[str, Any], forced_hw: Optional[tuple[int, int]] = None):
    signature = inspect.signature(model.forward)
    sample = payload.get('sample') or {}
    task = str(payload.get('task') or '').lower()
    root_name, leaf = first_input_seed_module(model)
    args = []
    kwargs = {}
    input_refs = []

    for name, param in signature.parameters.items():
        lower = name.lower()
        if name == 'self':
            continue
        if param.default is not inspect._empty and not any(
            token in lower for token in ('x', 'input', 'image', 'img', 'pixel', 'tokens', 'ids', 'mask', 'sample')
        ):
            kwargs[name] = param.default
            continue

        value = None
        input_strategy = 'synthesized'

        # ── 텍스트 리스트 파라미터 감지 (ingredient_texts, captions, sentences 등) ──
        if any(token in lower for token in ('ingredient', '_texts', 'captions', 'sentences', 'queries', 'prompts')):
            value = [['dummy ingredient a', 'dummy ingredient b']]
            input_strategy = 'text-list'
        elif any(token in lower for token in ('text', 'caption', 'sentence', 'query', 'prompt')) and 'mask' not in lower:
            value = ['dummy text']
            input_strategy = 'text-list'
        elif isinstance(leaf, nn.Embedding) or any(token in lower for token in ('ids', 'tokens', 'input_ids')):
            value = torch.randint(0, int(getattr(leaf, 'num_embeddings', 1000)), (1, 16), dtype=torch.long)
            input_strategy = 'token-sequence'
        elif any(token in lower for token in ('mask', 'attention_mask')):
            value = torch.ones((1, 16), dtype=torch.long)
            input_strategy = 'mask-sequence'
        elif isinstance(leaf, nn.Linear) and not any(token in task for token in ('image', 'vision', 'segmentation', 'classification')):
            value = torch.randn(1, int(getattr(leaf, 'in_features', 64)))
            input_strategy = 'linear-root'
        else:
            channels = 3
            if isinstance(leaf, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                channels = int(getattr(leaf, 'in_channels', 3))
            hw = image_hw_from_payload(payload, model, forced_hw=forced_hw)
            sample_path = sample.get('resolvedPath')
            if sample_path and os.path.exists(sample_path) and any(token in lower for token in ('x', 'image', 'img', 'pixel', 'input', 'sample', 'samples')):
                value = load_image_tensor(sample_path, channels, hw)
                input_strategy = 'sample-image'
            elif isinstance(leaf, nn.Conv1d):
                value = torch.randn(1, channels, 16000)
                input_strategy = 'conv1d-waveform'
            elif isinstance(leaf, nn.Conv3d):
                value = torch.randn(1, channels, 4, max(hw[0], 32), max(hw[1], 32))
                input_strategy = 'conv3d-video'
            else:
                value = torch.randn(1, channels, hw[0], hw[1])
                input_strategy = 'conv2d-image'

        if value is None:
            value = synth_value_for_param(param, model=model, payload=payload)

        input_refs.append({
            'name': name,
            'strategy': input_strategy,
        })

        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            args.append(value)
        else:
            kwargs[name] = value

    return args, kwargs, input_refs, root_name, leaf


def execute_trace(model: nn.Module, args: list[Any], kwargs: dict[str, Any]):
    collector = RuntimeTraceCollector()
    signature = inspect.signature(model.forward)
    collector.register_forward_inputs(args, kwargs, signature)

    handles = []
    for name, submodule in model.named_modules():
        if not name:
            continue
        module_is_leaf = is_leaf_module(submodule)
        def pre_hook(_module, _inputs, module_name=name, is_leaf=module_is_leaf):
            collector.current_module_stack.append(module_name)
            collector.current_leaf_stack.append(is_leaf)
            if is_leaf:
                collector.active_leaf_depth += 1
        def post_hook(_module, inputs, outputs, module_name=name, module_ref=submodule, is_leaf=module_is_leaf):
            collector._record_module(module_name, module_ref, inputs, outputs)
            collector.current_module_stack.pop()
            collector.current_leaf_stack.pop()
            if is_leaf:
                collector.active_leaf_depth -= 1
        handles.append(submodule.register_forward_pre_hook(pre_hook))
        handles.append(submodule.register_forward_hook(post_hook))

    try:
        started = time.time()
        with torch.no_grad():
            with collector.patched_functional_ops():
                with collector:
                    result = model(*args, **kwargs)
        runtime_ms = int((time.time() - started) * 1000)
    finally:
        for handle in handles:
            handle.remove()

    return_outputs = collector.refs_for(result, 'return')
    for ref in return_outputs:
        collector.return_outputs.append({
            'token': ref['token'],
            'label': ref['label'],
            'shape': ref.get('shape', ''),
        })

    return collector, runtime_ms


def maybe_retry_spatial_trace(model: nn.Module, payload: dict[str, Any], error: Exception):
    message = str(error).lower()
    if (
        'mat1 and mat2 shapes cannot be multiplied' not in message and
        'input size' not in message and
        'size mismatch' not in message and
        'invalid for input of size' not in message
    ):
        raise error

    tried = set()
    for hw in ((28, 28), (32, 32), (64, 64), (96, 96), (128, 128), (160, 160), (224, 224), (256, 256), (299, 299)):
        if hw in tried:
            continue
        tried.add(hw)
        try:
            args, kwargs, input_meta, _, _ = infer_forward_inputs(model, payload, forced_hw=hw)
            collector, runtime_ms = execute_trace(model, args, kwargs)
            strategies = ', '.join(sorted({item['strategy'] for item in input_meta if item.get('strategy')}))
            return collector, runtime_ms, strategies, input_meta
        except Exception:
            continue
    raise error


class RuntimeTraceCollector(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops: list[dict[str, Any]] = []
        self.return_outputs: list[dict[str, str]] = []
        self.forward_inputs: list[dict[str, str]] = []
        self.tensor_tokens: dict[int, str] = {}
        self.tensor_shapes: dict[str, str] = {}
        self.current_module_stack: list[str] = []
        self.current_leaf_stack: list[bool] = []
        self.active_leaf_depth = 0
        self.op_counter = 0
        self.token_counter = 0
        self.root_model_name = ''
        self.root_module_registered = False
        self.suspend_dispatch_recording = 0

    def next_token(self, label: str, tensor: torch.Tensor):
        token = f't{self.token_counter}'
        self.token_counter += 1
        self.tensor_tokens[id(tensor)] = token
        self.tensor_shapes[token] = str(list(tensor.shape))
        return {
            'token': token,
            'label': label,
            'shape': str(list(tensor.shape)),
        }

    def refs_for(self, value: Any, fallback_label: str = 'helper', allocate_new: bool = False) -> list[dict[str, str]]:
        refs = []
        for path, tensor in flatten_tensor_tree(value):
            token = None if allocate_new else self.tensor_tokens.get(id(tensor))
            label = path if path and path != 'tensor' else fallback_label
            if not token:
                ref = self.next_token(label, tensor)
            else:
                ref = {
                    'token': token,
                    'label': label,
                    'shape': self.tensor_shapes.get(token, ''),
                }
            refs.append(ref)
        return refs

    def register_forward_inputs(self, args: list[Any], kwargs: dict[str, Any], signature: inspect.Signature):
        ordered_names = [
            name for name in signature.parameters.keys()
            if name != 'self'
        ]
        arg_values = list(args)
        for idx, value in enumerate(arg_values):
            name = ordered_names[idx] if idx < len(ordered_names) else f'arg{idx}'
            refs = self.refs_for(value, name)
            for ref in refs:
                ref['label'] = name if ref['label'] in ('tensor', 'helper') else ref['label']
                self.forward_inputs.append(ref)
        for name, value in kwargs.items():
            refs = self.refs_for(value, name)
            for ref in refs:
                ref['label'] = name if ref['label'] in ('tensor', 'helper') else ref['label']
                self.forward_inputs.append(ref)

    def _record_module(self, module_name: str, module: nn.Module, inputs: Any, outputs: Any):
        if module_name == '':
            return
        if not is_leaf_module(module):
            return
        input_refs = self.refs_for(inputs, module_name)
        output_refs = self.refs_for(outputs, module_name, allocate_new=True)
        params = infer_simple_params(module)
        for idx, output_ref in enumerate(output_refs):
            label = module_name.split('.')[-1]
            if len(output_refs) > 1:
                label = f'{label}[{idx}]'
            self.ops.append({
                'id': f'op{self.op_counter}',
                'kind': 'module',
                'label': label,
                'inputs': copy.deepcopy(input_refs),
                'output': {
                    'token': output_ref['token'],
                    'label': label,
                    'shape': output_ref.get('shape', ''),
                },
                'block_type': normalize_block_type(module.__class__.__name__),
                'attr': module_name.split('.')[-1],
                'attr_path': module_name,
                'module_class': module.__class__.__name__,
                'params': params,
                'output_shape': output_ref.get('shape', ''),
            })
            self.op_counter += 1

    def _record_functional_call(self, label: str, args: tuple[Any, ...], kwargs: dict[str, Any], invoke):
        should_record = self.active_leaf_depth == 0
        if not should_record:
            return invoke()

        input_refs = self.refs_for((args, kwargs), label)
        try:
            self.suspend_dispatch_recording += 1
            result = invoke()
        finally:
            self.suspend_dispatch_recording -= 1

        output_refs = self.refs_for(result, label, allocate_new=True)
        block_type = normalize_block_type(label, FUNC_BLOCK_MAP.get(label.lower()))
        for idx, output_ref in enumerate(output_refs):
            op_label = label if len(output_refs) == 1 else f'{label}[{idx}]'
            self.ops.append({
                'id': f'op{self.op_counter}',
                'kind': 'op',
                'label': op_label,
                'inputs': copy.deepcopy(input_refs),
                'output': {
                    'token': output_ref['token'],
                    'label': op_label,
                    'shape': output_ref.get('shape', ''),
                },
                'block_type': block_type,
                'attr': None,
                'attr_path': self.current_module_stack[-1] if self.current_module_stack else None,
                'module_class': label,
                'params': {},
                'output_shape': output_ref.get('shape', ''),
            })
            self.op_counter += 1
        return result

    def patched_functional_ops(self):
        collector = self

        class FunctionalPatchContext:
            def __enter__(self_inner):
                self_inner.originals = []
                patched = [
                    (torch, 'flatten', 'flatten'),
                    (torch, 'reshape', 'reshape'),
                    (torch, 'permute', 'permute'),
                    (torch, 'transpose', 'transpose'),
                    (torch, 'cat', 'concat'),
                    (torch, 'stack', 'stack'),
                    (torch.Tensor, 'flatten', 'flatten'),
                    (torch.Tensor, 'reshape', 'reshape'),
                    (torch.Tensor, 'view', 'view'),
                    (torch.Tensor, 'permute', 'permute'),
                    (torch.Tensor, 'transpose', 'transpose'),
                    (torch.Tensor, '__getitem__', 'select'),
                ]

                for owner, attr, label in patched:
                    original = getattr(owner, attr, None)
                    if original is None:
                        continue

                    def make_wrapper(original_fn, call_label):
                        def wrapped(*args, **kwargs):
                            return collector._record_functional_call(
                                call_label,
                                args,
                                kwargs,
                                lambda: original_fn(*args, **kwargs),
                            )
                        return wrapped

                    setattr(owner, attr, make_wrapper(original, label))
                    self_inner.originals.append((owner, attr, original))
                return collector

            def __exit__(self_inner, exc_type, exc, tb):
                for owner, attr, original in reversed(self_inner.originals):
                    setattr(owner, attr, original)
                return False

        return FunctionalPatchContext()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Record only top-level tensor ops. Leaf module boundaries are already
        # captured by forward hooks, and duplicating their internal aten ops
        # creates dead-end artifacts in the rendered graph.
        should_record = self.active_leaf_depth == 0 and self.suspend_dispatch_recording == 0
        input_refs = self.refs_for((args, kwargs), func.__name__) if should_record else []
        result = func(*args, **kwargs)

        if should_record:
            output_refs = self.refs_for(result, func.__name__, allocate_new=True)
            op_name = getattr(getattr(func, 'overloadpacket', None), '__name__', None) or getattr(func, '__name__', 'op')
            block_type = normalize_block_type(op_name, FUNC_BLOCK_MAP.get(op_name.lower()))
            for idx, output_ref in enumerate(output_refs):
                label = op_name if len(output_refs) == 1 else f'{op_name}[{idx}]'
                self.ops.append({
                    'id': f'op{self.op_counter}',
                    'kind': 'op',
                    'label': label,
                    'inputs': copy.deepcopy(input_refs),
                    'output': {
                        'token': output_ref['token'],
                        'label': label,
                        'shape': output_ref.get('shape', ''),
                    },
                    'block_type': block_type,
                    'attr': None,
                    'attr_path': self.current_module_stack[-1] if self.current_module_stack else None,
                    'module_class': op_name,
                    'params': {},
                    'output_shape': output_ref.get('shape', ''),
                })
                self.op_counter += 1

        return result


def trace_model(payload: dict[str, Any]):
    repo_root = payload['repoRoot']
    source_file = payload['sourceFile']
    target_name = payload['modelName']
    preferred_factory = payload.get('runtimeFactory')

    module, import_meta = import_module_from_source(repo_root, source_file)
    model, constructor_meta = resolve_model_instance(module, target_name, preferred_factory, payload)
    args, kwargs, input_meta, _, _ = infer_forward_inputs(model, payload)
    try:
        collector, runtime_ms = execute_trace(model, args, kwargs)
        input_strategy = ', '.join(sorted({item['strategy'] for item in input_meta if item.get('strategy')}))
    except Exception as exc:
        collector, runtime_ms, input_strategy, input_meta = maybe_retry_spatial_trace(model, payload, exc)

    return {
        'success': True,
        'exactness': 'runtime_exact',
        'traceMode': 'runtime-exact',
        'importStrategy': import_meta.get('strategy', ''),
        'constructorStrategy': constructor_meta['kind'],
        'constructorCallable': constructor_meta['callable'],
        'inputStrategy': input_strategy,
        'inputEvidence': input_meta,
        'runtimeMs': runtime_ms,
        'model': {
            'name': model.__class__.__name__,
            'layers': [],
            'forward_order': [],
            'forward_inputs': collector.forward_inputs,
            'forward_graph': _prepend_preprocess_ops(
                collector.ops,
                _discover_transforms(model, payload),
                collector.forward_inputs,
                *image_hw_from_payload(payload, model),
                _get_input_channels(model),
                not any(ref.get('strategy') == 'sample-image' for ref in input_meta),
            ),
            'return_outputs': collector.return_outputs,
        },
    }


def main():
    payload = json.load(sys.stdin)
    try:
        result = trace_model(payload)
        to_json(result)
    except Exception as exc:
        to_json({
            'success': False,
            'exactness': 'unsupported_runtime_behavior',
            'traceMode': 'unsupported',
            'unsupportedReason': str(exc),
        })


if __name__ == '__main__':
    main()

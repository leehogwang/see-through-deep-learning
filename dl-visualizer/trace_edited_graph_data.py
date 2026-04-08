#!/usr/bin/env python3
import contextlib
import io
import json
import sys
from typing import Any, Optional

import torch
import torch.nn as nn

from runtime_trace import (
    import_module_from_source,
    resolve_model_instance,
    infer_forward_inputs,
    flatten_tensor_tree,
)
from trace_layer_data import tensor_to_preview


def _first_tensor(args: list, kwargs: dict) -> Optional[torch.Tensor]:
    for arg in args:
        flat = [tensor for _, tensor in flatten_tensor_tree(arg) if isinstance(tensor, torch.Tensor)]
        if flat:
            return flat[0]
    for _name, value in kwargs.items():
        flat = [tensor for _, tensor in flatten_tensor_tree(value) if isinstance(tensor, torch.Tensor)]
        if flat:
            return flat[0]
    return None


def _capture_outputs(model: nn.Module, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, outputs):
            flat = [tensor for _, tensor in flatten_tensor_tree(outputs) if isinstance(tensor, torch.Tensor)]
            if flat:
                captured[name] = flat[0]
        return hook

    for name, module in model.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        for handle in handles:
            handle.remove()

    return captured


def main():
    try:
        payload = json.loads(sys.stdin.read())
    except Exception as exc:
        json.dump({'previews': {}, 'inputPreview': None, 'error': f'invalid payload: {exc}'}, sys.stdout)
        return

    repo_root = payload.get('repoRoot', '')
    source_file = payload.get('sourceFile', '')
    model_name = payload.get('modelName', '')
    code = payload.get('code', '')
    alias_map = payload.get('aliasMap', {}) or {}

    if not repo_root or not source_file or not model_name or not code:
        json.dump({'previews': {}, 'inputPreview': None, 'error': 'repoRoot, sourceFile, modelName, code are required'}, sys.stdout)
        return

    try:
        module, _ = import_module_from_source(repo_root, source_file)
        original_model, _ = resolve_model_instance(module, model_name, payload.get('runtimeFactory'), payload)
        original_model.eval()
        args, kwargs, _input_meta, _root_name, _leaf = infer_forward_inputs(original_model, payload)
        input_tensor = _first_tensor(args, kwargs)
        if input_tensor is None:
            raise RuntimeError('no tensor input inferred from original model')
    except Exception as exc:
        json.dump({'previews': {}, 'inputPreview': None, 'error': f'input inference failed: {exc}'}, sys.stdout)
        return

    globals_dict: dict[str, Any] = {'torch': torch, 'nn': nn}
    locals_dict: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, globals_dict, locals_dict)
        generated_model = locals_dict.get('model') or globals_dict.get('model')
        if generated_model is None:
            generated_cls = locals_dict.get('GeneratedModel') or globals_dict.get('GeneratedModel')
            if generated_cls is None:
                raise RuntimeError('GeneratedModel not found in code')
            generated_model = generated_cls()
        generated_model.eval()
    except Exception as exc:
        json.dump({'previews': {}, 'inputPreview': None, 'error': f'generated graph build failed: {exc}'}, sys.stdout)
        return

    try:
        outputs = _capture_outputs(generated_model, input_tensor)
    except Exception as exc:
        json.dump({'previews': {}, 'inputPreview': None, 'error': f'generated graph forward failed: {exc}'}, sys.stdout)
        return

    previews: dict[str, Any] = {}
    for node_id, alias in alias_map.items():
        tensor = outputs.get(alias)
        if tensor is None:
            continue
        preview = tensor_to_preview(tensor)
        if preview is not None:
            previews[node_id] = preview

    input_preview = tensor_to_preview(input_tensor)
    json.dump({'previews': previews, 'inputPreview': input_preview, 'error': None}, sys.stdout)


if __name__ == '__main__':
    main()
import ast
import glob
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any


def to_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload))


def safe_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def parse_python_defaults(file_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    constants: dict[str, Any] = {}
    arg_defaults: dict[str, Any] = {}
    tree = ast.parse(file_path.read_text(encoding='utf-8'), filename=str(file_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = safe_literal(node.value)
            if value is not None:
                constants[node.targets[0].id] = value

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            option_name = None
            for arg in node.args:
                value = safe_literal(arg)
                if isinstance(value, str) and value.startswith('--'):
                    option_name = value[2:].replace('-', '_')
                    break
            if not option_name:
                continue
            for kw in node.keywords:
                if kw.arg == 'default':
                    default_value = safe_literal(kw.value)
                    if default_value is not None:
                        arg_defaults[option_name] = default_value
                    break

    return constants, arg_defaults


def resolve_path(value: Any, repo_root: Path, source_dir: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)

    source_candidate = (source_dir / value).resolve()
    if source_candidate.exists():
        return str(source_candidate)

    repo_candidate = (repo_root / value).resolve()
    return str(repo_candidate)


def merge_config(repo_root: Path, source_file: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    source_dir = source_file.parent
    candidates = [source_file]
    for pattern in ('train*.py', 'evaluate*.py', 'eval*.py', 'gradcam*.py', 'dataset*.py'):
        candidates.extend(sorted(source_dir.glob(pattern)))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            constants, arg_defaults = parse_python_defaults(candidate)
            merged.update(constants)
            merged.update(arg_defaults)
        except Exception:
            continue

    resolved: dict[str, Any] = {}
    for key, value in merged.items():
        resolved[key] = resolve_path(value, repo_root, source_dir)
    return resolved


def load_dataset_classes(source_dir: Path) -> list[type]:
    dataset_classes: list[type] = []
    sys.path.insert(0, str(source_dir))
    try:
        from torch.utils.data import Dataset  # type: ignore
    except Exception:
        return dataset_classes

    for file_path in sorted(source_dir.glob('dataset*.py')):
        module_name = file_path.stem
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            try:
                if issubclass(obj, Dataset) and obj is not Dataset:
                    dataset_classes.append(obj)
            except Exception:
                continue
    return dataset_classes


def build_dataset_kwargs(dataset_cls: type, config: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(dataset_cls.__init__)
    kwargs: dict[str, Any] = {}
    metadata_cafe1 = config.get('metadata_cafe1') or config.get('METADATA_CAFE1')
    metadata_cafe2 = config.get('metadata_cafe2') or config.get('METADATA_CAFE2')

    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if name == 'metadata_paths' and metadata_cafe1 and metadata_cafe2:
            kwargs[name] = [metadata_cafe1, metadata_cafe2]
        elif name == 'imagery_root':
            value = config.get('imagery_root') or config.get('IMAGERY_ROOT')
            if value:
                kwargs[name] = value
        elif name == 'split':
            kwargs[name] = 'test'
        elif name == 'split_json_path':
            value = config.get('split_json') or config.get('split_json_path')
            if value:
                kwargs[name] = value
        elif name in config:
            kwargs[name] = config[name]
        elif param.default is inspect._empty:
            raise ValueError(f'missing required dataset arg: {name}')

    if 'enable_ingredient_augmentation' in sig.parameters and 'enable_ingredient_augmentation' not in kwargs:
        kwargs['enable_ingredient_augmentation'] = False
    return kwargs


def extract_first_sample_path(dataset: Any) -> str:
    samples = getattr(dataset, 'samples', None)
    if not samples:
        raise ValueError('dataset has no samples')
    first = samples[0]
    if isinstance(first, tuple) and first:
        sample_path = first[0]
    else:
        sample_path = first
    if not isinstance(sample_path, str) or not sample_path:
        raise ValueError('first dataset sample path is invalid')
    return sample_path


def discover_nutrition5k_sample(config: dict[str, Any]) -> dict[str, Any] | None:
    metadata_cafe1 = config.get('metadata_cafe1') or config.get('METADATA_CAFE1')
    metadata_cafe2 = config.get('metadata_cafe2') or config.get('METADATA_CAFE2')
    imagery_root = config.get('imagery_root') or config.get('IMAGERY_ROOT')
    splits_dir = config.get('splits_dir')
    if not metadata_cafe1 or not metadata_cafe2 or not imagery_root or not splits_dir:
        return None

    split_file = Path(splits_dir) / 'rgb_test_ids.txt'
    if not split_file.exists():
        return None

    with split_file.open('r', encoding='utf-8') as handle:
        test_dish_ids = {line.strip() for line in handle if line.strip()}

    metadata_rows: list[list[str]] = []
    for csv_path in (metadata_cafe1, metadata_cafe2):
        csv_file = Path(csv_path)
        if not csv_file.exists():
            continue
        with csv_file.open('r', encoding='utf-8') as handle:
            for line in handle:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    metadata_rows.append(parts)

    for row in metadata_rows:
        dish_id = row[0]
        if dish_id not in test_dish_ids:
            continue
        frames_dir = Path(imagery_root) / dish_id / 'frames_sampled5'
        frame_paths = sorted(glob.glob(str(frames_dir / '*.jpeg')))
        if frame_paths:
            return {
                'resolvedPath': frame_paths[0],
                'strategy': 'static-dataset-logic',
                'source': 'dataset',
                'datasetClass': 'Nutrition5kDataset',
                'evidence': 'Reconstructed from metadata CSVs, split files, and imagery_root using dataset.py logic.',
            }

    return None


def main() -> None:
    payload = json.load(sys.stdin)
    repo_root = Path(payload['repoRoot']).resolve()
    source_file = Path(payload['sourceFile']).resolve()
    source_dir = source_file.parent

    try:
        config = merge_config(repo_root, source_file)
        dataset_classes = load_dataset_classes(source_dir)
        errors: list[str] = []

        for dataset_cls in dataset_classes:
            try:
                kwargs = build_dataset_kwargs(dataset_cls, config)
                dataset = dataset_cls(**kwargs)
                sample_path = extract_first_sample_path(dataset)
                if not os.path.exists(sample_path):
                    raise FileNotFoundError(sample_path)
                to_json({
                    'success': True,
                    'resolvedPath': sample_path,
                    'strategy': 'dataset-first-sample',
                    'source': 'dataset',
                    'datasetClass': dataset_cls.__name__,
                    'evidence': f'{dataset_cls.__name__}.samples[0] resolved from actual dataset construction.',
                })
                return
            except Exception as exc:
                errors.append(f'{dataset_cls.__name__}: {exc}')

        static_result = discover_nutrition5k_sample(config)
        if static_result:
            to_json({
                'success': True,
                **static_result,
            })
            return

        to_json({
            'success': False,
            'error': 'sample discovery failed',
            'details': errors[:10],
        })
    except Exception as exc:
        to_json({
            'success': False,
            'error': str(exc),
        })


if __name__ == '__main__':
    main()
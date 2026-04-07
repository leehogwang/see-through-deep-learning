#!/usr/bin/env python3
"""
Parse PyTorch nn.Module classes from a Python file.
Handles:
- nn.* layers
- imported sub-modules
- conditional branches / ternary assignments
- forward() dataflow graph extraction
"""
import ast
import copy
import sys
import json
import os

NN_TYPE_MAP = {
    'Conv1d': 'Conv1D', 'Conv2d': 'Conv2D', 'Conv3d': 'Conv3D',
    'ConvTranspose2d': 'TransposedConv2D',
    'MaxPool1d': 'MaxPool2D', 'MaxPool2d': 'MaxPool2D', 'MaxPool3d': 'MaxPool2D',
    'AvgPool1d': 'AvgPool2D', 'AvgPool2d': 'AvgPool2D',
    'AdaptiveAvgPool2d': 'AdaptiveAvgPool', 'AdaptiveMaxPool2d': 'GlobalMaxPool',
    'ReLU': 'ReLU', 'LeakyReLU': 'LeakyReLU', 'PReLU': 'PReLU',
    'ELU': 'ELU', 'SELU': 'SELU', 'GELU': 'GELU', 'SiLU': 'SiLU',
    'Mish': 'Mish', 'Sigmoid': 'Sigmoid', 'Tanh': 'Tanh',
    'Softmax': 'Softmax', 'Hardswish': 'HardSwish',
    'BatchNorm1d': 'BatchNorm2D', 'BatchNorm2d': 'BatchNorm2D',
    'LayerNorm': 'LayerNorm', 'GroupNorm': 'GroupNorm',
    'InstanceNorm2d': 'InstanceNorm',
    'MultiheadAttention': 'MultiHeadAttention',
    'LSTM': 'LSTM', 'GRU': 'GRU', 'RNN': 'RNN',
    'Embedding': 'Embedding',
    'Linear': 'Linear', 'Bilinear': 'Bilinear',
    'Dropout': 'Dropout', 'Dropout2d': 'Dropout2D',
    'Flatten': 'Flatten',
    'Sequential': 'Sequential',
    'TransformerEncoderLayer': 'TransformerEncoderLayer',
    'TransformerDecoderLayer': 'TransformerDecoderLayer',
    'Parameter': None,  # skip
}

FUNC_BLOCK_MAP = {
    'relu': 'ReLU',
    'leaky_relu': 'LeakyReLU',
    'gelu': 'GELU',
    'sigmoid': 'Sigmoid',
    'tanh': 'Tanh',
    'softmax': 'Softmax',
    'adaptive_avg_pool2d': 'AdaptiveAvgPool',
    'flatten': 'Flatten',
    'reshape': 'Reshape',
    'view': 'Reshape',
    'permute': 'Permute',
    'transpose': 'Permute',
    'cat': 'Concat',
    'concat': 'Concat',
    'add': 'Add',
}

IGNORE_CALLS = {
    'len', 'next', 'bool', 'isinstance', 'print',
}

IGNORE_TENSOR_INIT_CALLS = {
    'zeros', 'empty', 'zeros_like', 'empty_like',
}

PASSTHROUGH_METHODS = {
    'to', 'float', 'cuda', 'cpu', 'detach', 'contiguous',
}

SCALAR_METHODS = {
    'any', 'all', 'sum', 'item', 'size', 'dim', 'parameters',
}


def const_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = const_value(node.operand)
        if isinstance(v, (int, float, complex)):
            return -v
        raw = safe_unparse(node)
        return raw if raw is not None else None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        try:
            return ast.unparse(node)
        except Exception:
            return None
    return None


def call_to_params(call_node):
    params = {}
    for i, arg in enumerate(call_node.args):
        v = const_value(arg)
        if v is not None:
            params[f'arg{i}'] = v
    for kw in call_node.keywords:
        if kw.arg:
            v = const_value(kw.value)
            if v is not None:
                params[kw.arg] = v
    return params


def safe_unparse(node):
    try:
        return ast.unparse(node)
    except Exception:
        return None


def self_attr_path(node):
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name) and cur.id == 'self' and parts:
        return '.'.join(reversed(parts))
    return None


def unique_refs(refs):
    seen = set()
    out = []
    for ref in refs:
        token = ref['token']
        if token in seen:
            continue
        seen.add(token)
        out.append(ref)
    return out


def is_nn_module_base(base):
    if isinstance(base, ast.Attribute):
        return base.attr == 'Module'
    if isinstance(base, ast.Name):
        return base.id in ('Module', 'nn.Module')
    return False


def get_nn_type(call_node):
    """(mapped_type, nn_name) or (None, None)."""
    if not isinstance(call_node, ast.Call):
        return None, None
    func = call_node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in ('nn', 'torch.nn'):
            name = func.attr
            if name in NN_TYPE_MAP and NN_TYPE_MAP[name] is None:
                return None, None
            return (NN_TYPE_MAP.get(name) or name), name
    if isinstance(func, ast.Name) and func.id in NN_TYPE_MAP:
        name = func.id
        if NN_TYPE_MAP[name] is None:
            return None, None
        return (NN_TYPE_MAP.get(name) or name), name
    return None, None


def get_module_class_name(call_node, known_classes):
    """Return class name if this call is a known module subclass."""
    if not isinstance(call_node, ast.Call):
        return None
    func = call_node.func
    if isinstance(func, ast.Name) and func.id in known_classes:
        return func.id
    return None


def collect_imports(tree):
    """Collect all names imported from other modules (potential sub-module classes)."""
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported.add(name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported.add(name)
    return imported


def is_likely_module_class(name, imported_names):
    """Heuristic: capitalized name that's imported or known."""
    if not name or not name[0].isupper():
        return False
    skip = {'True', 'False', 'None', 'DataLoader', 'Tensor', 'Module'}
    if name in skip:
        return False
    return name in imported_names


def extract_call_from_expr(expr):
    """
    Given an expr, try to get the actual nn.Module call out of it.
    Handles direct calls and ternary assignments.
    """
    if isinstance(expr, ast.Call):
        return expr
    if isinstance(expr, ast.IfExp):
        for candidate in [expr.body, expr.orelse]:
            if isinstance(candidate, ast.Call):
                return candidate
            if isinstance(candidate, ast.Constant) and candidate.value is None:
                continue
    return None


def collect_self_assignments(init_node, known_classes, imported_names):
    """
    Walk the entire __init__ body (including if/else branches)
    and collect all self.xxx = <module>(...) assignments.
    Returns dict: attr -> layer_info
    """
    layers = {}

    def register_layer(attr, info):
        existing = layers.get(attr)
        if not existing:
            layers[attr] = info
            return

        existing_kind = existing.get('nn_type') or existing.get('type')
        new_kind = info.get('nn_type') or info.get('type')
        variants = existing.setdefault('variants', [existing_kind])
        if new_kind not in variants:
            variants.append(new_kind)

        existing_params = existing.setdefault('params', {})
        existing_params['variants'] = ' | '.join(variants)
        existing['label'] = f"{attr} ({' | '.join(variants)})"
        existing['is_custom'] = True

    def process_assign(target, value):
        if not (isinstance(target, ast.Attribute) and
                isinstance(target.value, ast.Name) and
                target.value.id == 'self'):
            return
        attr = target.attr

        call = extract_call_from_expr(value)
        if call is None:
            return

        mapped, nn_name = get_nn_type(call)
        if mapped and nn_name:
            params = call_to_params(call)
            register_layer(attr, {
                'type': mapped,
                'nn_type': nn_name,
                'params': params,
                'label': attr,
                'attr': attr,
                'is_custom': False,
            })
            return

        cname = get_module_class_name(call, known_classes)
        if cname:
            params = call_to_params(call)
            register_layer(attr, {
                'type': cname,
                'nn_type': cname,
                'params': params,
                'label': f'{attr} ({cname})',
                'attr': attr,
                'is_custom': True,
            })
            return

        if isinstance(call.func, ast.Name) and is_likely_module_class(call.func.id, imported_names):
            cname = call.func.id
            params = call_to_params(call)
            register_layer(attr, {
                'type': cname,
                'nn_type': cname,
                'params': params,
                'label': f'{attr} ({cname})',
                'attr': attr,
                'is_custom': True,
            })

    def walk_stmts(stmts):
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    process_assign(target, stmt.value)
            elif isinstance(stmt, ast.AnnAssign) and stmt.value:
                process_assign(stmt.target, stmt.value)
            elif isinstance(stmt, ast.If):
                walk_stmts(stmt.body)
                walk_stmts(stmt.orelse)
            elif isinstance(stmt, (ast.For, ast.While)):
                walk_stmts(stmt.body)
            elif isinstance(stmt, ast.With):
                walk_stmts(stmt.body)
            elif isinstance(stmt, ast.Try):
                walk_stmts(stmt.body)
                for handler in stmt.handlers:
                    walk_stmts(handler.body)

    walk_stmts(init_node.body)
    return layers


def collect_self_calls_in_order(stmts, layer_names):
    """Walk forward statements, collecting self.X refs innermost-first."""
    order = []
    seen = set()

    def visit_expr(node):
        if isinstance(node, ast.Call):
            for arg in node.args:
                visit_expr(arg)
            for kw in node.keywords:
                visit_expr(kw.value)
            if isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name) and
                        node.func.value.id == 'self' and
                        node.func.attr in layer_names):
                    name = node.func.attr
                    if name not in seen:
                        order.append(name)
                        seen.add(name)
                else:
                    visit_expr(node.func.value)
        elif isinstance(node, ast.Attribute):
            if (isinstance(node.value, ast.Name) and
                    node.value.id == 'self' and
                    node.attr in layer_names):
                if node.attr not in seen:
                    order.append(node.attr)
                    seen.add(node.attr)

    def visit_stmt(stmt):
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.expr):
                visit_expr(child)
            elif isinstance(child, ast.stmt):
                visit_stmt(child)

    for stmt in stmts:
        visit_stmt(stmt)

    return order


def target_name(target):
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Subscript):
        return target_name(target.value)
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        return f'{target.value.id}.{target.attr}'
    return None


def func_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def lower_func_name(func):
    name = func_name(func)
    return name.lower() if name else None


def make_call_params(node):
    return call_to_params(node) if isinstance(node, ast.Call) else {}


class ForwardGraphBuilder:
    def __init__(self, layers):
        self.layers = layers
        self.ops = []
        self.forward_inputs = []
        self.return_outputs = []
        self.op_idx = 0
        self.token_idx = 0

    def make_ref(self, label):
        ref = {
            'token': f'v{self.token_idx}_{label}',
            'label': label,
        }
        self.token_idx += 1
        return ref

    def make_temp_ref(self):
        return self.make_ref(f'__tmp{self.token_idx}')

    def add_input(self, name, env):
        ref = self.make_ref(name)
        self.forward_inputs.append(ref)
        env[name] = [ref]

    def emit_op(self, kind, label, inputs, output_ref, block_type=None, attr=None, module_class=None, params=None, attr_path=None):
        op = {
            'id': f'op_{self.op_idx}',
            'kind': kind,
            'label': label,
            'inputs': unique_refs(inputs),
            'output': output_ref,
            'block_type': block_type,
            'attr': attr,
            'attr_path': attr_path,
            'module_class': module_class,
            'params': params or {},
        }
        self.op_idx += 1
        self.ops.append(op)
        return [output_ref] if output_ref else []

    def same_refs(self, refs_a, refs_b):
        tokens_a = [ref['token'] for ref in unique_refs(refs_a or [])]
        tokens_b = [ref['token'] for ref in unique_refs(refs_b or [])]
        return tokens_a == tokens_b

    def merge_refs(self, name, refs_a, refs_b, guard=None):
        refs_a = unique_refs(copy.deepcopy(refs_a or []))
        refs_b = unique_refs(copy.deepcopy(refs_b or []))

        if not refs_a:
            return refs_b
        if not refs_b:
            return refs_a
        if self.same_refs(refs_a, refs_b):
            return refs_a

        output_ref = self.make_ref(name)
        params = {'branch': 'conditional'}
        if guard:
            params['guard'] = guard
        self.emit_op(
            kind='op',
            label='merge',
            inputs=refs_a + refs_b,
            output_ref=output_ref,
            block_type=None,
            params=params,
        )
        return [output_ref]

    def merge_envs(self, base_env, env_a, env_b, guard=None):
        merged = {}
        all_keys = set(base_env.keys()) | set(env_a.keys()) | set(env_b.keys())
        for key in all_keys:
            refs_a = env_a.get(key, base_env.get(key, []))
            refs_b = env_b.get(key, base_env.get(key, []))
            merged[key] = self.merge_refs(key, refs_a, refs_b, guard=guard)
        return merged

    def is_scalarish_expr(self, node):
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.is_scalarish_expr(node.operand)
        if isinstance(node, ast.BoolOp):
            return all(self.is_scalarish_expr(value) for value in node.values)
        if isinstance(node, ast.Compare):
            if all(isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)) for op in node.ops):
                return True
            operands = [node.left, *node.comparators]
            return all(self.is_scalarish_expr(operand) for operand in operands)
        if isinstance(node, ast.Attribute):
            return node.attr in {'device', 'dtype', 'shape'}
        if isinstance(node, ast.Call):
            lname = lower_func_name(node.func)
            if lname in {name.lower() for name in IGNORE_CALLS}:
                return True
            if lname in SCALAR_METHODS:
                return True
        return False

    def resolve_name(self, name, env, non_tensor_vars):
        if name in non_tensor_vars:
            return []
        return copy.deepcopy(env.get(name, []))

    def bind_pattern_refs(self, target, refs, env):
        refs = unique_refs(copy.deepcopy(refs or []))
        if isinstance(target, ast.Name):
            env[target.id] = refs
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self.bind_pattern_refs(elt, refs, env)

    def comprehension_refs(self, node, env, non_tensor_vars, desired_label=None):
        comp_env = copy.deepcopy(env)
        refs = []

        for gen in node.generators:
            iter_refs = self.expr_refs(gen.iter, comp_env, non_tensor_vars)
            refs.extend(iter_refs)
            self.bind_pattern_refs(gen.target, iter_refs, comp_env)
            for if_clause in gen.ifs:
                refs.extend(self.expr_refs(if_clause, comp_env, non_tensor_vars))

        if isinstance(node, ast.DictComp):
            refs.extend(self.expr_refs(node.key, comp_env, non_tensor_vars))
            refs.extend(self.expr_refs(node.value, comp_env, non_tensor_vars, desired_label))
        else:
            refs.extend(self.expr_refs(node.elt, comp_env, non_tensor_vars, desired_label))

        return unique_refs(refs)

    def expr_refs(self, node, env, non_tensor_vars, desired_label=None):
        if node is None:
            return []

        if isinstance(node, ast.Name):
            return self.resolve_name(node.id, env, non_tensor_vars)

        if isinstance(node, ast.Subscript):
            base_refs = self.expr_refs(node.value, env, non_tensor_vars)
            slice_refs = self.expr_refs(node.slice, env, non_tensor_vars)
            selector = safe_unparse(node.slice)
            if not base_refs and slice_refs:
                return unique_refs(slice_refs)
            if not base_refs:
                return []
            output_ref = self.make_ref(desired_label) if desired_label else self.make_temp_ref()
            return self.emit_op(
                kind='op',
                label='select',
                inputs=base_refs + slice_refs,
                output_ref=output_ref,
                block_type='Select',
                params={'selector': selector} if selector else {},
            )

        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'self':
                return []
            if node.attr in {'device', 'dtype', 'shape'}:
                return []
            return self.expr_refs(node.value, env, non_tensor_vars)

        if isinstance(node, ast.Tuple):
            refs = []
            for elt in node.elts:
                refs.extend(self.expr_refs(elt, env, non_tensor_vars))
            return unique_refs(refs)

        if isinstance(node, ast.List):
            refs = []
            for elt in node.elts:
                refs.extend(self.expr_refs(elt, env, non_tensor_vars))
            return unique_refs(refs)

        if isinstance(node, ast.Dict):
            refs = []
            for key, value in zip(node.keys, node.values):
                label = desired_label
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    label = key.value
                refs.extend(self.expr_refs(value, env, non_tensor_vars, desired_label=label))
            return unique_refs(refs)

        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            return self.comprehension_refs(node, env, non_tensor_vars, desired_label)

        if isinstance(node, ast.IfExp):
            refs = []
            refs.extend(self.expr_refs(node.body, env, non_tensor_vars, desired_label))
            refs.extend(self.expr_refs(node.orelse, env, non_tensor_vars, desired_label))
            return unique_refs(refs)

        if isinstance(node, ast.BoolOp):
            refs = []
            for value in node.values:
                refs.extend(self.expr_refs(value, env, non_tensor_vars, desired_label))
            return unique_refs(refs)

        if isinstance(node, ast.Compare):
            refs = []
            refs.extend(self.expr_refs(node.left, env, non_tensor_vars, desired_label))
            for comparator in node.comparators:
                refs.extend(self.expr_refs(comparator, env, non_tensor_vars, desired_label))
            return unique_refs(refs)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return self.expr_refs(node.operand, env, non_tensor_vars, desired_label)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.expr_refs(node.operand, env, non_tensor_vars, desired_label)

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            inputs = self.collect_add_inputs(node, env, non_tensor_vars)
            output_ref = self.make_ref(desired_label) if desired_label else self.make_temp_ref()
            return self.emit_op(
                kind='op',
                label='Add',
                inputs=inputs,
                output_ref=output_ref,
                block_type='Add',
                params={},
            )

        if isinstance(node, ast.Call):
            return self.call_refs(node, env, non_tensor_vars, desired_label)

        return []

    def collect_add_inputs(self, node, env, non_tensor_vars):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            refs = []
            refs.extend(self.collect_add_inputs(node.left, env, non_tensor_vars))
            refs.extend(self.collect_add_inputs(node.right, env, non_tensor_vars))
            return unique_refs(refs)
        return self.expr_refs(node, env, non_tensor_vars)

    def call_refs(self, node, env, non_tensor_vars, desired_label=None):
        lname = lower_func_name(node.func)

        # Ignore scalar-ish helpers entirely
        if lname in {name.lower() for name in IGNORE_CALLS}:
            return []
        if lname in SCALAR_METHODS:
            return []

        # Pass-through tensor methods (x.to(...), x.float(), ...)
        if isinstance(node.func, ast.Attribute) and node.func.attr in PASSTHROUGH_METHODS:
            return self.expr_refs(node.func.value, env, non_tensor_vars)

        # Preserve tensor init helpers as placeholder/default nodes
        if lname in IGNORE_TENSOR_INIT_CALLS:
            output_ref = self.make_ref(desired_label) if desired_label else self.make_temp_ref()
            return self.emit_op(
                kind='op',
                label=lname,
                inputs=[],
                output_ref=output_ref,
                block_type='Placeholder',
                params=make_call_params(node),
            )

        # self.module(...)
        if isinstance(node.func, ast.Attribute):
            full_attr = self_attr_path(node.func)
            if full_attr:
                attr = full_attr.split('.')[-1]
                inputs = []
                for arg in node.args:
                    inputs.extend(self.expr_refs(arg, env, non_tensor_vars))
                for kw in node.keywords:
                    inputs.extend(self.expr_refs(kw.value, env, non_tensor_vars))
                inputs = unique_refs(inputs)

                layer = self.layers.get(attr)
                output_ref = self.make_ref(desired_label) if desired_label else self.make_temp_ref()
                if layer and full_attr == attr:
                    block_type = layer['type'] if layer['type'] in NN_TYPE_MAP.values() else None
                    module_class = layer.get('nn_type') or layer.get('type')
                    label = attr
                    return self.emit_op(
                        kind='module',
                        label=label,
                        inputs=inputs,
                        output_ref=output_ref,
                        block_type=block_type,
                        attr=attr,
                        attr_path=full_attr,
                        module_class=module_class,
                        params=layer.get('params') or {},
                    )

                return self.emit_op(
                    kind='module',
                    label=attr,
                    inputs=inputs,
                    output_ref=output_ref,
                    block_type=None,
                    attr=attr,
                    attr_path=full_attr,
                    module_class=full_attr,
                    params=make_call_params(node),
                )

        # Tensor method call: x.view(...), x.mean(...), x.expand_as(...)
        if isinstance(node.func, ast.Attribute):
            is_namespace_call = isinstance(node.func.value, ast.Name) and node.func.value.id in {'F', 'torch', 'nn'}
            receiver_refs = self.expr_refs(node.func.value, env, non_tensor_vars)
            is_bound_tensor_method = receiver_refs or not isinstance(node.func.value, ast.Name)
            if not is_namespace_call and is_bound_tensor_method:
                method = node.func.attr
                inputs = list(receiver_refs)
                if method in {'expand_as'}:
                    if node.args:
                        inputs.extend(self.expr_refs(node.args[0], env, non_tensor_vars))
                elif method in {'cat', 'concat'}:
                    for arg in node.args:
                        inputs.extend(self.expr_refs(arg, env, non_tensor_vars))
                output_ref = self.make_ref(desired_label) if desired_label else self.make_temp_ref()
                block_type = FUNC_BLOCK_MAP.get(method.lower())
                return self.emit_op(
                    kind='op',
                    label=method,
                    inputs=inputs,
                    output_ref=output_ref,
                    block_type=block_type,
                    params=make_call_params(node),
                )

        # F.relu(...), torch.stack(...), bare fn(...)
        inputs = []
        for arg in node.args:
            inputs.extend(self.expr_refs(arg, env, non_tensor_vars))
        for kw in node.keywords:
            inputs.extend(self.expr_refs(kw.value, env, non_tensor_vars))
        inputs = unique_refs(inputs)

        if not inputs and node.args:
            first_arg = target_name(node.args[0])
            if first_arg and first_arg in env:
                inputs = copy.deepcopy(env[first_arg])

        raw_name = func_name(node.func) or 'call'
        block_type = FUNC_BLOCK_MAP.get(raw_name.lower())
        output_ref = self.make_ref(desired_label) if desired_label else self.make_temp_ref()
        return self.emit_op(
            kind='op',
            label=raw_name,
            inputs=inputs,
            output_ref=output_ref,
            block_type=block_type,
            params=make_call_params(node),
        )

    def assign_target(self, name, refs, env, merge=False):
        refs = unique_refs(refs)
        if merge and name in env:
            env[name] = unique_refs(env[name] + refs)
        else:
            env[name] = refs

    def walk_stmts(self, stmts, env, non_tensor_vars):
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    name = target_name(target)
                    if name:
                        if self.is_scalarish_expr(stmt.value):
                            non_tensor_vars.add(name)
                            env[name] = []
                            continue
                        refs = self.expr_refs(stmt.value, env, non_tensor_vars, desired_label=name)
                        if refs:
                            merge = isinstance(target, ast.Subscript)
                            if merge and name in env:
                                selector = safe_unparse(target.slice)
                                refs = self.merge_refs(
                                    name,
                                    env.get(name, []),
                                    refs,
                                    guard=f'partial update: {selector}' if selector else 'partial update',
                                )
                                self.assign_target(name, refs, env)
                            else:
                                self.assign_target(name, refs, env, merge=merge)
                        continue
                    if isinstance(target, (ast.Tuple, ast.List)):
                        refs = self.expr_refs(stmt.value, env, non_tensor_vars)
                        if refs:
                            self.bind_pattern_refs(target, refs, env)
                        continue
                # fallback: still walk value for nested ops
                self.expr_refs(stmt.value, env, non_tensor_vars)

            elif isinstance(stmt, ast.AnnAssign) and stmt.value:
                name = target_name(stmt.target)
                if name:
                    if self.is_scalarish_expr(stmt.value):
                        non_tensor_vars.add(name)
                        env[name] = []
                        continue
                    refs = self.expr_refs(stmt.value, env, non_tensor_vars, desired_label=name)
                    if refs:
                        self.assign_target(name, refs, env)
                        continue

            elif isinstance(stmt, ast.Expr):
                # list.append(expr) → treat expr result as feeding that list variable
                if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute):
                    method = stmt.value.func.attr
                    base = stmt.value.func.value
                    if method == 'append' and isinstance(base, ast.Name) and stmt.value.args:
                        list_name = base.id
                        refs = self.expr_refs(stmt.value.args[0], env, non_tensor_vars, desired_label=list_name)
                        if refs:
                            self.assign_target(list_name, refs, env, merge=True)
                        continue
                self.expr_refs(stmt.value, env, non_tensor_vars)

            elif isinstance(stmt, ast.If):
                env_then = copy.deepcopy(env)
                env_else = copy.deepcopy(env)
                non_tensor_then = set(non_tensor_vars)
                non_tensor_else = set(non_tensor_vars)
                self.walk_stmts(stmt.body, env_then, non_tensor_then)
                self.walk_stmts(stmt.orelse, env_else, non_tensor_else)
                merged = self.merge_envs(env, env_then, env_else, guard=safe_unparse(stmt.test))
                env.clear()
                env.update(merged)
                non_tensor_vars.clear()
                non_tensor_vars.update(non_tensor_then | non_tensor_else)

            elif isinstance(stmt, ast.For):
                loop_env = copy.deepcopy(env)
                loop_non_tensor = set(non_tensor_vars)
                iter_refs = self.expr_refs(stmt.iter, env, non_tensor_vars)
                self.bind_pattern_refs(stmt.target, iter_refs, loop_env)
                self.walk_stmts(stmt.body, loop_env, loop_non_tensor)
                merged = self.merge_envs(env, loop_env, env)
                env.clear()
                env.update(merged)
                non_tensor_vars.clear()
                non_tensor_vars.update(loop_non_tensor)

            elif isinstance(stmt, ast.While):
                loop_env = copy.deepcopy(env)
                loop_non_tensor = set(non_tensor_vars)
                self.walk_stmts(stmt.body, loop_env, loop_non_tensor)
                merged = self.merge_envs(env, loop_env, env)
                env.clear()
                env.update(merged)
                non_tensor_vars.clear()
                non_tensor_vars.update(loop_non_tensor)

            elif isinstance(stmt, ast.With):
                self.walk_stmts(stmt.body, env, non_tensor_vars)

            elif isinstance(stmt, ast.Try):
                body_env = copy.deepcopy(env)
                else_env = copy.deepcopy(env)
                body_non_tensor = set(non_tensor_vars)
                else_non_tensor = set(non_tensor_vars)
                self.walk_stmts(stmt.body, body_env, body_non_tensor)
                for handler in stmt.handlers:
                    handler_env = copy.deepcopy(env)
                    handler_non_tensor = set(non_tensor_vars)
                    self.walk_stmts(handler.body, handler_env, handler_non_tensor)
                    body_env = self.merge_envs(body_env, body_env, handler_env)
                    body_non_tensor |= handler_non_tensor
                self.walk_stmts(stmt.orelse, else_env, else_non_tensor)
                merged = self.merge_envs(env, body_env, else_env)
                env.clear()
                env.update(merged)
                non_tensor_vars.clear()
                non_tensor_vars.update(body_non_tensor | else_non_tensor)

            elif isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Dict):
                    outputs = []
                    for key, value in zip(stmt.value.keys, stmt.value.values):
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            label = key.value
                        else:
                            label = 'output'
                        refs = self.expr_refs(value, env, non_tensor_vars, desired_label=label)
                        outputs.extend(refs)
                    self.return_outputs = unique_refs(self.return_outputs + outputs)
                else:
                    refs = self.expr_refs(stmt.value, env, non_tensor_vars, desired_label='return')
                    self.return_outputs = unique_refs(self.return_outputs + refs)

    def build(self, forward_node):
        env = {}
        non_tensor_vars = set()
        for arg in forward_node.args.args[1:]:  # skip self
            self.add_input(arg.arg, env)
        self.walk_stmts(forward_node.body, env, non_tensor_vars)
        return {
            'forward_inputs': self.forward_inputs,
            'forward_graph': self.ops,
            'return_outputs': self.return_outputs,
        }


def parse_class(class_node, known_classes, imported_names):
    layers = {}
    forward_order = []
    forward_graph = {
        'forward_inputs': [],
        'forward_graph': [],
        'return_outputs': [],
    }

    for item in class_node.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        if item.name == '__init__':
            layers = collect_self_assignments(item, known_classes, imported_names)
        elif item.name == 'forward':
            forward_order = collect_self_calls_in_order(item.body, set(layers.keys()))
            builder = ForwardGraphBuilder(layers)
            forward_graph = builder.build(item)

    ordered = []
    seen = set()
    for name in forward_order:
        if name in layers:
            ordered.append(layers[name])
            seen.add(name)
    for name, info in layers.items():
        if name not in seen:
            ordered.append(info)

    return {
        'name': class_node.name,
        'layers': ordered,
        'forward_order': forward_order,
        'forward_inputs': forward_graph['forward_inputs'],
        'forward_graph': forward_graph['forward_graph'],
        'return_outputs': forward_graph['return_outputs'],
    }


def parse_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
    except Exception as e:
        return {'error': str(e), 'models': []}

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return {'error': f'SyntaxError: {e}', 'models': []}

    local_classes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(is_nn_module_base(b) for b in node.bases):
                local_classes.add(node.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in local_classes:
                    local_classes.add(node.name)

    imported_names = collect_imports(tree)

    models = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(is_nn_module_base(b) for b in node.bases):
                models.append(parse_class(node, local_classes, imported_names))

    return {'models': models, 'file': filepath}


def scan_directory(dirpath):
    results = []
    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fname in files:
            if not fname.endswith('.py'):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'nn.Module' not in content and 'torch.nn.Module' not in content:
                    continue
                tree = ast.parse(content, filename=fpath)
                model_classes = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if any(is_nn_module_base(b) for b in node.bases):
                            model_classes.append(node.name)
                if model_classes:
                    results.append({
                        'file': fpath,
                        'relative': os.path.relpath(fpath, dirpath),
                        'models': model_classes,
                    })
            except Exception:
                continue
    return results


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Usage: parse_model.py <scan|parse> <path>'}))
        sys.exit(1)
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == 'scan':
        print(json.dumps(scan_directory(path)))
    elif cmd == 'parse':
        print(json.dumps(parse_file(path)))
    else:
        print(json.dumps({'error': f'Unknown command: {cmd}'}))

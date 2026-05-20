import argparse
import json
import os
import re
from typing import Dict

import torch


ADAPTER_NAME = 'adapter_model.bin'
EMA_NAME = 'router_distill_ema.bin'
MODULE_SPECS = (
    ('wq', 'lora_attn_wq', 'attn_wq_weight'),
    ('wk', 'lora_attn_wk', 'attn_wk_weight'),
    ('proto', 'lora_attn_expert_proto', 'attn_expert_proto'),
)


def _normalize_module_name(name: str) -> str:
    prefixes = ('base_model.model.', 'base_model.', 'model.')
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


def _load_torch_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location='cpu')


def _extract_student_state(adapter_dir: str) -> Dict[str, Dict[str, torch.Tensor]]:
    state = _load_torch_file(os.path.join(adapter_dir, ADAPTER_NAME))
    modules: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        for kind, token_name, _ in MODULE_SPECS:
            marker = f'.{token_name}'
            if marker not in key:
                continue
            prefix, suffix = key.split(marker, 1)
            if suffix and not suffix.startswith('.'):
                continue
            module_name = _normalize_module_name(prefix)
            modules.setdefault(module_name, {})[kind] = tensor.detach().float().cpu()
            break
    return modules


def _extract_ema_state(adapter_dir: str) -> Dict[str, Dict[str, torch.Tensor]]:
    path = os.path.join(adapter_dir, EMA_NAME)
    if not os.path.exists(path):
        return {}
    state = _load_torch_file(path)
    modules: Dict[str, Dict[str, torch.Tensor]] = {}
    for module_name, module_state in state.items():
        module_name = _normalize_module_name(module_name)
        dst = modules.setdefault(module_name, {})
        for kind, _, ema_name in MODULE_SPECS:
            tensor = module_state.get(ema_name)
            if isinstance(tensor, torch.Tensor):
                dst[kind] = tensor.detach().float().cpu()
    return modules


def _compare_groups(
    lhs: Dict[str, Dict[str, torch.Tensor]],
    rhs: Dict[str, Dict[str, torch.Tensor]],
    lhs_label: str,
    rhs_label: str,
    topn: int,
):
    eps = 1e-12
    rows = []
    for module_name in sorted(set(lhs) & set(rhs)):
        for kind, _, _ in MODULE_SPECS:
            lhs_tensor = lhs[module_name].get(kind)
            rhs_tensor = rhs[module_name].get(kind)
            if lhs_tensor is None or rhs_tensor is None:
                continue
            if lhs_tensor.shape != rhs_tensor.shape:
                continue
            diff = rhs_tensor - lhs_tensor
            lhs_norm = lhs_tensor.norm().item()
            rhs_norm = rhs_tensor.norm().item()
            diff_norm = diff.norm().item()
            rel_change = diff_norm / max(lhs_norm, eps)
            rows.append({
                'module': module_name,
                'kind': kind,
                'lhs_norm': lhs_norm,
                'rhs_norm': rhs_norm,
                'diff_norm': diff_norm,
                'rel_change': rel_change,
            })

    summary = {}
    for kind, _, _ in MODULE_SPECS:
        kind_rows = [row for row in rows if row['kind'] == kind]
        if not kind_rows:
            continue
        summary[kind] = {
            'count': len(kind_rows),
            'mean_diff_norm': sum(row['diff_norm'] for row in kind_rows) / len(kind_rows),
            'mean_rel_change': sum(row['rel_change'] for row in kind_rows) / len(kind_rows),
            'max_rel_change': max(row['rel_change'] for row in kind_rows),
            'mean_lhs_norm': sum(row['lhs_norm'] for row in kind_rows) / len(kind_rows),
            'mean_rhs_norm': sum(row['rhs_norm'] for row in kind_rows) / len(kind_rows),
            'top_modules': sorted(kind_rows, key=lambda row: row['rel_change'], reverse=True)[:topn],
        }

    return {
        'lhs_label': lhs_label,
        'rhs_label': rhs_label,
        'total_pairs': len(rows),
        'rows': rows,
        'summary': summary,
    }


def _print_report(report: dict):
    print(f"\n=== {report['lhs_label']} -> {report['rhs_label']} ===")
    print(f"total compared tensors: {report['total_pairs']}")
    if not report['summary']:
        print('no overlapping router tensors found')
        return

    for kind in ('wq', 'wk', 'proto'):
        stats = report['summary'].get(kind)
        if not stats:
            continue
        print(
            f"[{kind}] count={stats['count']} "
            f"mean_diff={stats['mean_diff_norm']:.6f} "
            f"mean_rel={stats['mean_rel_change']:.6f} "
            f"max_rel={stats['max_rel_change']:.6f}"
        )
        for row in stats['top_modules']:
            print(
                f"  - {row['module']}: rel={row['rel_change']:.6f}, "
                f"diff={row['diff_norm']:.6f}, "
                f"{report['lhs_label']}_norm={row['lhs_norm']:.6f}, "
                f"{report['rhs_label']}_norm={row['rhs_norm']:.6f}"
            )


def _extract_layer_id(module_name: str):
    match = re.search(r'layers\.(\d+)', module_name)
    if match:
        return int(match.group(1))
    return None


def _sanitize_filename(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', text).strip('_') or 'report'


def _plot_report(report: dict, plot_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('warning: matplotlib is not available, skip plot generation')
        return

    os.makedirs(plot_dir, exist_ok=True)
    tag = _sanitize_filename(f"{report['lhs_label']}_to_{report['rhs_label']}")

    kinds = ['wq', 'wk', 'proto']
    mean_rel = [report['summary'].get(kind, {}).get('mean_rel_change', 0.0) for kind in kinds]
    max_rel = [report['summary'].get(kind, {}).get('max_rel_change', 0.0) for kind in kinds]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = list(range(len(kinds)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], mean_rel, width=width, label='mean_rel_change')
    ax.bar([i + width / 2 for i in x], max_rel, width=width, label='max_rel_change')
    ax.set_xticks(x)
    ax.set_xticklabels(kinds)
    ax.set_ylabel('relative change')
    ax.set_title(f"{report['lhs_label']} -> {report['rhs_label']}")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    summary_path = os.path.join(plot_dir, f'{tag}_summary.png')
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)
    print(f'plot saved to {summary_path}')

    layer_curves = {kind: {} for kind in kinds}
    for row in report['rows']:
        layer_id = _extract_layer_id(row['module'])
        if layer_id is None:
            continue
        layer_curves[row['kind']].setdefault(layer_id, []).append(row['rel_change'])

    if not any(layer_curves.values()):
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind in kinds:
        items = sorted(layer_curves[kind].items())
        if not items:
            continue
        xs = [layer for layer, _ in items]
        ys = [sum(values) / len(values) for _, values in items]
        ax.plot(xs, ys, marker='o', label=kind)
    ax.set_xlabel('layer')
    ax.set_ylabel('mean relative change')
    ax.set_title(f"Per-layer change: {report['lhs_label']} -> {report['rhs_label']}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    layer_path = os.path.join(plot_dir, f'{tag}_per_layer.png')
    fig.savefig(layer_path, dpi=200)
    plt.close(fig)
    print(f'plot saved to {layer_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Compare change magnitude of lora_attn_wq / lora_attn_wk / lora_attn_expert_proto.'
    )
    parser.add_argument('adapter_dir_a', help='checkpoint/adaptor dir A')
    parser.add_argument('adapter_dir_b', nargs='?', default=None, help='checkpoint/adaptor dir B')
    parser.add_argument('--topn', type=int, default=5, help='show top-N changed modules per tensor type')
    parser.add_argument('--json-output', default=None, help='optional path to save the numeric report as json')
    parser.add_argument('--plot-dir', default=None, help='optional directory to save summary and per-layer plots')
    args = parser.parse_args()

    reports = []

    student_a = _extract_student_state(args.adapter_dir_a)
    ema_a = _extract_ema_state(args.adapter_dir_a)

    if ema_a:
        reports.append(_compare_groups(ema_a, student_a, 'ema', 'student', args.topn))
    else:
        print(f'warning: {EMA_NAME} not found in {args.adapter_dir_a}')

    if args.adapter_dir_b is not None:
        student_b = _extract_student_state(args.adapter_dir_b)
        reports.append(_compare_groups(student_a, student_b, 'student_a', 'student_b', args.topn))

        ema_b = _extract_ema_state(args.adapter_dir_b)
        if ema_a and ema_b:
            reports.append(_compare_groups(ema_a, ema_b, 'ema_a', 'ema_b', args.topn))
        elif os.path.exists(os.path.join(args.adapter_dir_b, EMA_NAME)):
            print(f'warning: failed to compare EMA states between {args.adapter_dir_a} and {args.adapter_dir_b}')

    for report in reports:
        _print_report(report)
        if args.plot_dir:
            _plot_report(report, args.plot_dir)

    if args.json_output:
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2)
        print(f'\njson report saved to {args.json_output}')


if __name__ == '__main__':
    main()

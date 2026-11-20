"""Is there a GPU this process can actually compute on?

`torch.cuda.is_available()` alone is not enough on a shared cluster: a node
can list a GPU that the job was not allocated, have a driver too old for the
installed CUDA build, or be out of memory. `check` therefore runs a small
matrix multiply on the device and compares it with the CPU result.
"""

import os
import shutil
import socket
import subprocess
import time


class GpuUnavailable(RuntimeError):
    pass


def _nvidia_smi():
    """Driver version and per-GPU memory from nvidia-smi, if present."""
    if not shutil.which('nvidia-smi'):
        return None
    fields = 'index,name,driver_version,memory.total,memory.used'
    try:
        out = subprocess.run(
            ['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except (subprocess.SubprocessError, OSError) as e:
        return {'error': str(e)}
    rows = [dict(zip(fields.split(','), (v.strip() for v in line.split(',')), strict=True))
            for line in out.strip().splitlines()]
    return {'gpus': rows}


def report():
    """Everything relevant to "why is there no GPU?", as a plain dict."""
    info = {
        'host': socket.gethostname(),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'nvidia_smi': _nvidia_smi(),
    }
    try:
        import torch
    except ImportError:
        info['torch'] = None
        return info
    info['torch'] = torch.__version__
    info['torch_cuda_build'] = torch.version.cuda
    info['cuda_available'] = torch.cuda.is_available()
    info['mps_available'] = torch.backends.mps.is_available()
    info['devices'] = []
    if info['cuda_available']:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info['devices'].append({
                'index': i, 'name': p.name, 'capability': f'{p.major}.{p.minor}',
                'total_gb': round(total / 1e9, 2), 'free_gb': round(free / 1e9, 2)})
    return info


def _diagnose(info, backend):
    if info.get('torch') is None:
        return 'torch is not installed: run `uv sync --frozen --extra gpu`.'
    if backend == 'mps':
        return 'MPS is unavailable (needs Apple silicon and macOS 12.3+).'
    if info.get('torch_cuda_build') is None:
        return (f'torch {info["torch"]} is a CPU-only build. On Linux, uv should '
                'install the +cu126 build from uv.lock; check `uv sync --frozen --extra gpu`.')
    smi = info.get('nvidia_smi')
    if smi is None:
        return ('no nvidia-smi: this node has no NVIDIA driver. Are you on a GPU '
                'node, and did the job request a GPU (e.g. --gres=gpu:1)?')
    if 'error' in smi:
        return f'nvidia-smi failed: {smi["error"]}'
    if info.get('CUDA_VISIBLE_DEVICES') in ('', '-1'):
        return 'CUDA_VISIBLE_DEVICES hides every GPU: the scheduler allocated none.'
    driver = smi['gpus'][0]['driver_version'] if smi.get('gpus') else '?'
    return (f'the driver ({driver}) is present but CUDA is unusable: it may be '
            f'older than torch\'s CUDA {info["torch_cuda_build"]} build needs.')


def check(backend='cuda', min_memory_gb=0.0, size=2048):
    """Return (device, report) after proving the device computes correctly.

    backend: 'cuda', 'mps', or 'any' (cuda, then mps). 'cpu' always passes,
    for running GPU stages slowly on a laptop.
    Raises GpuUnavailable with a diagnosis otherwise.
    """
    info = report()
    if backend == 'cpu':
        return 'cpu', info
    order = {'cuda': ['cuda'], 'mps': ['mps'], 'any': ['cuda', 'mps']}[backend]
    device = next((b for b in order if info.get(f'{b}_available')), None)
    if device is None:
        raise GpuUnavailable(_diagnose(info, order[0]))

    if device == 'cuda':
        free = max(d['free_gb'] for d in info['devices'])
        if free < min_memory_gb:
            raise GpuUnavailable(f'largest free GPU memory is {free} GB, '
                                 f'the job needs {min_memory_gb} GB.')

    import torch
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(size, size, generator=gen)
    b = torch.randn(size, size, generator=gen)
    try:
        start = time.perf_counter()
        on_device = (a.to(device) @ b.to(device)).cpu()
        elapsed = time.perf_counter() - start
    except RuntimeError as e:  # e.g. "no kernel image is available for execution"
        raise GpuUnavailable(f'{device} is listed but failed to compute: {e}') from e
    error = (on_device - a @ b).abs().max().item() / (a @ b).abs().max().item()
    if not error < 1e-3:  # also catches NaN
        raise GpuUnavailable(f'{device} returned wrong results (relative error {error:.2g}).')
    info['test'] = {'device': device, 'size': size, 'seconds': round(elapsed, 4),
                    'relative_error': error}
    return device, info

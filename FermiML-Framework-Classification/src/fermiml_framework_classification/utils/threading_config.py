import os
from typing import Any, Dict, Optional

import threadpoolctl
import torch

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Keep a process-lifetime reference to the threadpoolctl limiter; the BLAS thread
# limit it applies is restored as soon as the object is garbage collected, so it
# must outlive configure_threads().
_THREADPOOL_LIMITER: Optional[threadpoolctl._ThreadpoolLimiter] = None


def resolve_n_threads() -> int:
    """Resolve the number of CPU cores this process should use.

    Prefers SLURM's per-task allocation when present, then the CPU affinity mask
    (what the OS actually permits, which on a cgroup-enforced cluster reflects the
    allocation), then the total CPU count.

    :return: The intended thread count, at least 1.
    :rtype: int
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            n_threads = int(slurm_cpus)
            if n_threads > 0:
                return n_threads
        except ValueError:
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - non-Linux platforms
        return max(1, os.cpu_count() or 1)


def configure_threads(n_threads: Optional[int] = None) -> int:
    """Configure all numerical backends to use ``n_threads`` cores at runtime.

    Unlike setting ``OMP_NUM_THREADS`` after numpy/torch are imported (a no-op,
    since OpenMP/MKL read those variables only at load time), this applies the
    limit live: ``torch.set_num_threads`` for the torch ATen thread pool and
    ``threadpoolctl`` for the already-loaded BLAS libraries (MKL/OpenBLAS). The
    environment variables are also set so any child process inherits the limit.

    :param n_threads: Number of threads to use; resolved from the SLURM/affinity
        environment when ``None``.
    :return: The thread count that was applied.
    :rtype: int
    """
    global _THREADPOOL_LIMITER
    if n_threads is None:
        n_threads = resolve_n_threads()
    n_threads = max(1, int(n_threads))

    for env_var in THREAD_ENV_VARS:
        os.environ[env_var] = str(n_threads)

    torch.set_num_threads(n_threads)
    # Pin the inter-op pool too. Left at its default it stays at the machine's
    # total core count (e.g. 192 on a full node) even when only a handful of
    # cores are allocated, which oversubscribes the cgroup and intermittently
    # deadlocks torch inter-op work. set_num_interop_threads is one-shot per
    # process and raises once the pool has been used, so only call it when the
    # value actually needs to change and swallow the "already started" error.
    if torch.get_num_interop_threads() != n_threads:
        try:
            torch.set_num_interop_threads(n_threads)
        except RuntimeError:
            pass
    _THREADPOOL_LIMITER = threadpoolctl.threadpool_limits(limits=n_threads)
    return n_threads


def thread_report() -> Dict[str, Any]:
    """Collect a diagnostic snapshot of how many cores each backend will use.

    :return: A dictionary describing the allocation (CPU count, affinity size,
        ``SLURM_CPUS_PER_TASK``, the thread-count environment variables), the
        torch thread pools and the per-library BLAS thread counts reported by
        ``threadpoolctl``.
    :rtype: Dict[str, Any]
    """
    try:
        n_affinity: Optional[int] = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux platforms
        n_affinity = None

    return {
        "cpu_count": os.cpu_count(),
        "cpu_affinity": n_affinity,
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "env": {env_var: os.environ.get(env_var) for env_var in THREAD_ENV_VARS},
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "blas_threadpools": [
            {"prefix": pool.get("prefix"), "num_threads": pool.get("num_threads")}
            for pool in threadpoolctl.threadpool_info()
        ],
    }

import os
from unittest import mock

import torch

from fermiml_framework_classification.utils import threading_config
from fermiml_framework_classification.utils.threading_config import (
    THREAD_ENV_VARS,
    configure_threads,
    resolve_n_threads,
    thread_report,
)


class TestThreadingConfig:
    def _save_state(self):
        return (
            os.environ.get("SLURM_CPUS_PER_TASK"),
            {var: os.environ.get(var) for var in THREAD_ENV_VARS},
            torch.get_num_threads(),
        )

    def _restore_state(self, state):
        slurm_value, env_values, torch_threads = state
        if slurm_value is None:
            os.environ.pop("SLURM_CPUS_PER_TASK", None)
        else:
            os.environ["SLURM_CPUS_PER_TASK"] = slurm_value
        for var, value in env_values.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        torch.set_num_threads(torch_threads)

    def test_resolve_n_threads_reads_slurm_cpus_per_task(self):
        state = self._save_state()
        try:
            os.environ["SLURM_CPUS_PER_TASK"] = "5"
            assert resolve_n_threads() == 5
        finally:
            self._restore_state(state)

    def test_resolve_n_threads_falls_back_to_affinity(self):
        state = self._save_state()
        try:
            os.environ.pop("SLURM_CPUS_PER_TASK", None)
            assert resolve_n_threads() == max(1, len(os.sched_getaffinity(0)))
        finally:
            self._restore_state(state)

    def test_resolve_n_threads_ignores_invalid_slurm_value(self):
        state = self._save_state()
        try:
            os.environ["SLURM_CPUS_PER_TASK"] = "not-an-int"
            assert resolve_n_threads() >= 1
        finally:
            self._restore_state(state)

    def test_configure_threads_applies_requested_count(self):
        state = self._save_state()
        try:
            applied = configure_threads(2)
            assert applied == 2
            assert torch.get_num_threads() == 2
            for var in THREAD_ENV_VARS:
                assert os.environ[var] == "2"
        finally:
            self._restore_state(state)

    def test_configure_threads_clamps_to_at_least_one(self):
        state = self._save_state()
        try:
            assert configure_threads(0) == 1
            assert torch.get_num_threads() == 1
        finally:
            self._restore_state(state)

    def test_configure_threads_resolves_from_environment_when_none(self):
        state = self._save_state()
        try:
            os.environ["SLURM_CPUS_PER_TASK"] = "3"
            assert configure_threads() == 3
        finally:
            self._restore_state(state)

    def test_configure_threads_pins_interop_threads(self):
        state = self._save_state()
        try:
            # Force a mismatch so configure_threads has to update the pool, and
            # patch the torch calls so the assertion is independent of the
            # process-global (one-shot) interop state under pytest-xdist.
            with (
                mock.patch.object(threading_config.torch, "get_num_interop_threads", return_value=999),
                mock.patch.object(threading_config.torch, "set_num_interop_threads") as set_interop,
            ):
                configure_threads(2)
            set_interop.assert_called_once_with(2)
        finally:
            self._restore_state(state)

    def test_configure_threads_swallows_interop_runtime_error(self):
        state = self._save_state()
        try:
            # Once the inter-op pool has been used, set_num_interop_threads raises
            # RuntimeError; configure_threads must ignore it and still apply the
            # rest of the thread configuration.
            with (
                mock.patch.object(threading_config.torch, "get_num_interop_threads", return_value=999),
                mock.patch.object(
                    threading_config.torch,
                    "set_num_interop_threads",
                    side_effect=RuntimeError("cannot set number of interop threads"),
                ),
            ):
                assert configure_threads(2) == 2
            assert torch.get_num_threads() == 2
        finally:
            self._restore_state(state)

    def test_thread_report_contains_expected_keys(self):
        state = self._save_state()
        try:
            configure_threads(1)
            report = thread_report()
            for key in ("cpu_count", "cpu_affinity", "env", "torch_num_threads", "blas_threadpools"):
                assert key in report
            assert set(report["env"].keys()) == set(THREAD_ENV_VARS)
        finally:
            self._restore_state(state)

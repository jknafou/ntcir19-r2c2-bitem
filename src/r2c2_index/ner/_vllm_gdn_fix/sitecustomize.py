"""Triton-allocator fix for Qwen3.5 gated-delta-net (GDN) on non-Hopper GPUs.

Why this exists
---------------
vLLM 0.17.1 routes the Qwen3.5 gated-delta-net forward through the FlashInfer
kernel only on compute capability exactly 9.0 (Hopper / H100 / H200):
``vllm/model_executor/models/qwen3_next.py:156`` does ``is_device_capability(90)``.
On every other CUDA capability (notably Blackwell sm_120, e.g. the RTX PRO
6000) it falls back to ``forward_native`` -> ``fla`` chunk_gated_delta_
rule, whose ``solve_tril`` kernel uses Triton ``make_tensor_descriptor`` (TMA)
and therefore needs a Triton global-scratch allocator. vLLM calls
``triton.set_allocator`` on its fused-MoE path alone, so a dense model like the
Qwen3.5-9B NER model fails on Blackwell with::

    RuntimeError: Kernel requires a runtime memory allocation, but no allocator
    was set. Use triton.set_allocator to specify an allocator.

The 122B agent model is MoE, so it takes the fused-MoE path that sets the
allocator, and runs on Blackwell without this patch.

Fix
---
Lazily install vLLM's own ``set_triton_allocator`` the first time the GDN forward
runs, in whatever process/thread actually executes the model (the v1 EngineCore
worker). The Triton allocator is a per-process ContextVar, so it must be set
inside the worker context, which is why the patch acts on the first call and
not at module load. It is a runtime patch: the vLLM source and the package
version are unchanged. It affects the dense Blackwell path alone; H100/H200
(FlashInfer path) and A100 (no TMA branch, sm_80) never call it.

How it activates
----------------
CPython auto-imports ``sitecustomize`` at interpreter startup whenever its
directory is on ``PYTHONPATH``, including in forked or spawned EngineCore
subprocesses. The NER worker sbatch puts this directory on ``PYTHONPATH`` before
launching vLLM. See ``src/r2c2_index/slurm/ner_worker.sbatch``.
"""

import os


def _install_gdn_allocator_patch() -> None:
    try:
        import torch
        import vllm.model_executor.models.qwen3_next as q
        from vllm.triton_utils.allocation import set_triton_allocator
    except Exception:
        # vLLM or torch is not importable in this interpreter (e.g. a helper
        # process), so there is nothing to patch.
        return

    cls = getattr(q, "ChunkGatedDeltaRule", None)
    if cls is None or getattr(cls, "_gdn_allocator_patched", False):
        return

    orig_forward_native = cls.forward_native

    def forward_native(self, *args, **kwargs):
        # Set the Triton allocator once, in this worker process/thread, before
        # the first solve_tril TMA kernel launch.
        if not getattr(self, "_gdn_alloc_set", False):
            try:
                dev = torch.device("cuda", torch.cuda.current_device())
            except Exception:
                dev = torch.device("cuda")
            set_triton_allocator(dev)
            self._gdn_alloc_set = True
            print(
                f"[gdn-allocator-fix] triton.set_allocator installed on {dev} "
                f"for ChunkGatedDeltaRule.forward_native (pid={os.getpid()})",
                flush=True,
            )
        return orig_forward_native(self, *args, **kwargs)

    cls.forward_native = forward_native
    cls._gdn_allocator_patched = True


_install_gdn_allocator_patch()

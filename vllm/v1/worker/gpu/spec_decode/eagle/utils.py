# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model


def _should_share(eagle: nn.Module, flag: str, draft, target) -> bool:
    """Share when the draft has no own copy, or its copy matches the target."""

    if not getattr(eagle, flag, False) or draft is None:
        return True
    if target is None:
        return False
    # torch.equal on GPU allocates a bool mask the size of the input.
    # Use the faster GPU path when there is plenty of headroom;
    # otherwise compare on CPU.
    w = draft.weight
    if w.is_cuda and torch.cuda.mem_get_info(w.device)[0] < w.numel() * 2:
        return torch.equal(w.cpu(), target.weight.cpu())
    return torch.equal(w, target.weight)


def create_eagle_draft_vllm_config(vllm_config: VllmConfig) -> VllmConfig:
    """Return a vLLM config scoped to the EAGLE draft model.

    The target model may force an attention backend that is only valid for the
    target architecture, e.g. B12X sparse MLA for GLM. EAGLE draft models can be
    non-sparse MLA models, so their requested draft_attention_backend must be
    applied before constructing draft attention layers and metadata builders.
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None

    config = replace(
        vllm_config,
        parallel_config=replace(
            speculative_config.draft_parallel_config,
            rank=vllm_config.parallel_config.rank,
        ),
        model_config=speculative_config.draft_model_config,
    )
    if speculative_config.moe_backend is not None:
        config = replace(
            config,
            kernel_config=replace(
                config.kernel_config,
                moe_backend=speculative_config.moe_backend,
            ),
        )
    if speculative_config.draft_kv_cache_dtype is not None:
        config = replace(
            config,
            cache_config=replace(
                config.cache_config,
                cache_dtype=speculative_config.draft_kv_cache_dtype,
            ),
        )
    if speculative_config.draft_attention_backend is not None:
        draft_backend = (
            None
            if speculative_config.draft_attention_backend == "auto"
            else speculative_config.draft_attention_backend
        )
        config = replace(
            config,
            attention_config=replace(
                config.attention_config,
                backend=draft_backend,
            ),
        )
    return config


def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    draft_vllm_config = create_eagle_draft_vllm_config(vllm_config)
    with set_model_tag("eagle_head"):
        eagle_model = get_model(
            vllm_config=draft_vllm_config,
            model_config=draft_model_config,
            load_config=speculative_config.draft_load_config,
            prefix="eagle_head",
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = eagle_model.model

    # Skip embedding sharing under PP — each rank owns its own embedding.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            eagle_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = getattr(target_model, "lm_head", None)
    draft_lm_head = getattr(eagle_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        eagle_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del eagle_model.lm_head
        eagle_model.lm_head = target_lm_head

        # MTP layers route logits through layer.shared_head.head, not
        # eagle_model.lm_head, so the per-layer copies need fixing up too.
        layers = getattr(draft_inner, "layers", None)
        if layers is not None:
            items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
            for layer in items:
                sh = getattr(layer, "shared_head", None)
                if sh is not None and hasattr(sh, "head"):
                    del sh.head
                    sh.head = target_lm_head

    # MTP also shares a topk_indices_buffer between target and draft.
    if hasattr(target_inner, "topk_indices_buffer"):
        if hasattr(draft_inner, "topk_indices_buffer"):
            del draft_inner.topk_indices_buffer
        draft_inner.topk_indices_buffer = target_inner.topk_indices_buffer

    return eagle_model

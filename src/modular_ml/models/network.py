"""ModularNet and the config-driven network factory.

``build_modular_model`` turns an :class:`ArchitectureConfig` into a ``ModularNet``
by Hydra-instantiating each block spec. The net runs its blocks in declared
order, threading a single ``DataDict`` through them. ``apply_runtime_schedule``
applies epoch-triggered freeze/unfreeze/input-mode changes.
"""

from typing import Any, Dict, List, Optional, Tuple

import hydra
from torch import nn

from modular_ml.configs.init_configs import InitSpecs
from modular_ml.configs.model_configs import BlockSpec, ModelConfig
from modular_ml.models.blocks import Block, DataDict


class ModularNet(nn.Module):
    """A sequence of :class:`Block` modules sharing one data dict."""

    def __init__(
        self,
        modules: List[Block],
        module_ids: List[str],
        module_init_specs: Optional[Dict[str, Dict[str, InitSpecs]]] = None,
        module_freeze_specs: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__()
        self.modules_seq = nn.ModuleList(modules)
        self.module_ids = module_ids
        self.module_map = dict(zip(module_ids, self.modules_seq))
        self.module_init_specs = module_init_specs or {}
        self.module_freeze_specs = module_freeze_specs or {}
        self._apply_init()
        self._apply_freeze()

    def _apply_init(self) -> None:
        for mid, module in self.module_map.items():
            specs = self.module_init_specs.get(mid)
            if specs and hasattr(module, "set_parameters"):
                module.set_parameters(specs)

    def _apply_freeze(self) -> None:
        for mid, module in self.module_map.items():
            freeze = self.module_freeze_specs.get(mid)
            if freeze and hasattr(module, "freeze"):
                module.freeze(freeze)

    def get_module(self, module_id: str) -> Block:
        """Return a block by id."""
        try:
            return self.module_map[module_id]
        except KeyError as exc:
            raise KeyError(f"Unknown module id '{module_id}'") from exc

    def apply_sparsity_masks(self, apply_grad: bool = True) -> None:
        """Re-apply every block's sparsity masks."""
        for module in self.modules_seq:
            if hasattr(module, "apply_sparsity_masks"):
                module.apply_sparsity_masks(apply_grad=apply_grad)

    def apply_runtime_schedule(self, stage_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply one runtime-schedule stage (per-module freeze/unfreeze/modes)."""
        if not stage_state:
            return {}
        module_updates = stage_state.get("modules", {})
        applied: Dict[str, Any] = {}
        for module_id, updates in module_updates.items():
            if not updates:
                continue
            module = self.get_module(module_id)
            done: Dict[str, Any] = {}
            modes = updates.get("input_modes")
            if modes is not None and hasattr(module, "set_input_modes"):
                module.set_input_modes(list(modes))
                done["input_modes"] = list(modes)
            freeze = updates.get("freeze")
            if freeze:
                module.freeze(list(freeze))
                done["freeze"] = list(freeze)
            unfreeze = updates.get("unfreeze")
            if unfreeze:
                module.unfreeze(list(unfreeze))
                done["unfreeze"] = list(unfreeze)
            if done:
                applied[module_id] = done
        return applied

    def forward(self, data_dict: DataDict) -> DataDict:
        """Run every block in order, threading the shared data dict."""
        for module in self.modules_seq:
            data_dict = module(data_dict)
        return data_dict


def _instantiate_block(
    spec: BlockSpec, idx: int
) -> Tuple[str, Block, Dict[str, InitSpecs], List[str]]:
    """Hydra-instantiate one block from its spec, flattening params/kwargs."""
    kwargs: Dict[str, Any] = {
        "_target_": spec._block_target_,
        "id": spec.id or f"block_{idx}",
        "in_keys": spec.in_keys,
        "out_keys": spec.out_keys,
    }
    kwargs.update(getattr(spec, "params", {}) or {})
    # Only pass in_dim/out_dim if the block expects them (params may override).
    if "in_dim" not in kwargs and "in_dims" not in kwargs and spec.in_dim:
        kwargs["in_dim"] = spec.in_dim
    if "out_dim" not in kwargs and spec.out_dim:
        kwargs["out_dim"] = spec.out_dim
    kwargs["init_specs"] = getattr(spec, "init", {}) or {}
    kwargs["freeze_specs"] = list(getattr(spec, "freeze", []) or [])
    module = hydra.utils.instantiate(kwargs)
    return kwargs["id"], module, kwargs["init_specs"], kwargs["freeze_specs"]


def build_modular_model(model_cfg: ModelConfig) -> ModularNet:
    """Build a ``ModularNet`` from a model config's architecture."""
    specs: List[BlockSpec] = list(model_cfg.architecture.modules.values())
    if not specs:
        raise ValueError("ArchitectureConfig must define at least one BlockSpec.")

    modules, ids = [], []
    init_specs: Dict[str, Dict[str, InitSpecs]] = {}
    freeze_specs: Dict[str, List[str]] = {}
    for idx, spec in enumerate(specs):
        mid, module, m_init, m_freeze = _instantiate_block(spec, idx)
        ids.append(mid)
        modules.append(module)
        init_specs[mid] = m_init
        freeze_specs[mid] = m_freeze
    return ModularNet(modules, ids, init_specs, freeze_specs)

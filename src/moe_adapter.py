#!/usr/bin/env python3
"""
Model-agnostic MoE adapter. Lets the same probes run on different MoE LLMs
(Qwen3-30B-A3B and Mixtral-8x7B) by auto-detecting, per decoder layer:
  - the MoE block module (Qwen: `layer.mlp`, Mixtral: `layer.block_sparse_moe`)
  - its `.gate` (router) and `.experts` (ModuleList)
  - num_experts and top_k

Usage:
  from src.moe_adapter import detect_moe
  info = detect_moe(model)
  info.blocks      # list of MoE block modules (one per MoE layer), in order
  info.gates       # list of gate modules (info.blocks[i].gate)
  info.num_layers, info.num_experts, info.top_k
  info.block_attr  # "mlp" or "block_sparse_moe"
"""
from dataclasses import dataclass, field
from typing import List
import torch.nn as nn

# Candidate attribute names for the MoE block on a decoder layer, in priority order.
_BLOCK_ATTRS = ["mlp", "block_sparse_moe", "moe", "feed_forward"]


@dataclass
class MoEInfo:
    block_attr: str
    blocks: List[nn.Module] = field(default_factory=list)
    gates: List[nn.Module] = field(default_factory=list)
    num_layers: int = 0
    num_experts: int = 0
    top_k: int = 0


def _decoder_layers(model):
    return model.model.layers if hasattr(model, "model") else model.layers


def _is_moe_block(mod) -> bool:
    return mod is not None and hasattr(mod, "gate") and hasattr(mod, "experts")


def _infer_top_k(model, block) -> int:
    cfg = getattr(model, "config", None)
    for name in ("num_experts_per_tok", "moe_topk", "top_k", "num_selected_experts"):
        v = getattr(cfg, name, None) if cfg else None
        if isinstance(v, int) and v > 0:
            return v
    # fallback for Qwen3-MoE default
    return 8


def detect_moe(model) -> MoEInfo:
    layers = _decoder_layers(model)
    block_attr = None
    for attr in _BLOCK_ATTRS:
        if any(_is_moe_block(getattr(l, attr, None)) for l in layers):
            block_attr = attr
            break
    if block_attr is None:
        raise RuntimeError("No MoE block (with .gate and .experts) found on any layer.")

    info = MoEInfo(block_attr=block_attr)
    for l in layers:
        b = getattr(l, attr := block_attr, None)
        if _is_moe_block(b):
            info.blocks.append(b)
            info.gates.append(b.gate)
    info.num_layers = len(info.blocks)
    info.num_experts = len(info.blocks[0].experts)
    info.top_k = _infer_top_k(model, info.blocks[0])
    return info

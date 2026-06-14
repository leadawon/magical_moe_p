"""
RoutingProbe: capture and aggregate MoE router behaviour via forward hooks.

We hook the `gate` Linear of every Qwen3MoeSparseMoeBlock. The gate output is
exactly the router logits of shape [num_tokens, num_experts]. From those logits
we reproduce the model's own top-k selection (softmax -> top-k) and accumulate,
per (layer, expert), the statistics needed to test the routing hypotheses.

Captured per layer (accumulated over a whole dataset):
  - sel_counts [L,E]   : # times each expert appears in a token's top-k
  - prob_mass  [L,E]   : sum of pre-selection softmax prob per expert
  - weight_mass[L,E]   : sum of renormalised top-k routing weight per expert
  - tokens     [L]     : number of tokens routed
  - top1_sum   [L]     : sum over tokens of the top-1 softmax prob (confidence)
  - topk_sum   [L]     : sum over tokens of cumulative top-k prob mass
  - entropy_sum[L]     : sum over tokens of routing-distribution entropy
  - margin_sum [L]     : sum over tokens of (prob[k-th] - prob[(k+1)-th])

Per-example "fingerprints" [N,L,E] (top-k selection frequency, rows sum to 1)
are also stored so examples can be clustered / classified by domain afterwards.

The hooks are framework-version independent: a gate is a plain nn.Linear, so its
forward output is the logits regardless of how the surrounding block is written.
"""

from typing import List
import numpy as np
import torch


class RoutingProbe:
    def __init__(self, model, num_experts: int = 128, top_k: int = 8):
        self.num_experts = num_experts
        self.top_k = top_k
        self.captured = {}          # layer_idx -> logits tensor (on its device)
        self.handles = []
        self.num_layers = self._register(model)
        self.reset_dataset()

    # ----- hook registration -------------------------------------------------
    def _register(self, model) -> int:
        # Model-agnostic: works for Qwen (layer.mlp) and Mixtral
        # (layer.block_sparse_moe) via the MoE adapter.
        from src.moe_adapter import detect_moe
        info = detect_moe(model)
        for moe_idx, gate in enumerate(info.gates):
            h = gate.register_forward_hook(self._make_hook(moe_idx))
            self.handles.append(h)
        if not self.handles:
            raise RuntimeError("No MoE gate modules found to hook.")
        return info.num_layers

    def _make_hook(self, idx: int):
        def hook(module, inp, out):
            # out: [num_tokens, num_experts]
            self.captured[idx] = out.detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    # ----- accumulators ------------------------------------------------------
    def reset_dataset(self):
        L, E = self.num_layers, self.num_experts
        self.sel_counts = torch.zeros(L, E, dtype=torch.float64)
        self.prob_mass = torch.zeros(L, E, dtype=torch.float64)
        self.weight_mass = torch.zeros(L, E, dtype=torch.float64)
        self.tokens = torch.zeros(L, dtype=torch.float64)
        self.top1_sum = torch.zeros(L, dtype=torch.float64)
        self.topk_sum = torch.zeros(L, dtype=torch.float64)
        self.entropy_sum = torch.zeros(L, dtype=torch.float64)
        self.margin_sum = torch.zeros(L, dtype=torch.float64)
        self.fingerprints: List[np.ndarray] = []

    # ----- per-example aggregation ------------------------------------------
    @torch.no_grad()
    def consume_example(self):
        """Aggregate the routing captured during the last forward pass."""
        L, E, k = self.num_layers, self.num_experts, self.top_k
        fp = np.zeros((L, E), dtype=np.float32)

        for l in range(L):
            logits = self.captured[l].float()          # [T, E] on its device
            probs = torch.softmax(logits, dim=-1)       # [T, E]
            T = probs.shape[0]

            topw, topi = torch.topk(probs, k, dim=-1)   # [T, k] sorted desc
            flat_i = topi.reshape(-1)

            counts = torch.bincount(flat_i, minlength=E).double()           # [E]
            self.sel_counts[l] += counts.cpu()
            self.prob_mass[l] += probs.sum(0).double().cpu()

            wn = topw / (topw.sum(-1, keepdim=True) + 1e-9)                  # renorm
            wmass = torch.zeros(E, device=probs.device, dtype=torch.float32)
            wmass.scatter_add_(0, flat_i, wn.reshape(-1))
            self.weight_mass[l] += wmass.double().cpu()

            self.tokens[l] += T
            self.top1_sum[l] += topw[:, 0].sum().item()
            self.topk_sum[l] += topw.sum(-1).sum().item()
            ent = -(probs * torch.log(probs + 1e-12)).sum(-1)
            self.entropy_sum[l] += ent.sum().item()

            top_kp1 = torch.topk(probs, k + 1, dim=-1).values               # [T, k+1]
            margin = top_kp1[:, k - 1] - top_kp1[:, k]
            self.margin_sum[l] += margin.sum().item()

            fp[l] = (counts / (T * k)).cpu().numpy()    # selection-freq distribution

        self.fingerprints.append(fp)
        self.captured.clear()

    # ----- export ------------------------------------------------------------
    def save(self, path: str, dataset: str, domain: str):
        fps = np.stack(self.fingerprints, axis=0) if self.fingerprints \
            else np.zeros((0, self.num_layers, self.num_experts), dtype=np.float32)
        np.savez_compressed(
            path,
            dataset=dataset,
            domain=domain,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            top_k=self.top_k,
            sel_counts=self.sel_counts.numpy(),
            prob_mass=self.prob_mass.numpy(),
            weight_mass=self.weight_mass.numpy(),
            tokens=self.tokens.numpy(),
            top1_sum=self.top1_sum.numpy(),
            topk_sum=self.topk_sum.numpy(),
            entropy_sum=self.entropy_sum.numpy(),
            margin_sum=self.margin_sum.numpy(),
            fingerprints=fps,
        )

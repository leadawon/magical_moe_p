"""
Dataset loaders for the MoE routing-probe experiment.

Six datasets across three domains:
  - math : gsm8k, svamp
  - code : humaneval_plus, mbpp_plus
  - nli  : mnli, snli

Each loader returns a list of dicts: {"id": str, "text": str}.

Design choice: we feed the *raw task text* (no chat template, no instruction
scaffolding) so that the captured routing reflects how the model processes
domain content, not shared chat/template tokens. The same minimal framing is
used within each domain so cross-domain comparison is apples-to-apples.
All loaders use a fixed seed so the sampled subset is reproducible.
"""

import random
from typing import List, Dict

from datasets import load_dataset

DOMAIN_OF = {
    "gsm8k": "math",
    "svamp": "math",
    "humaneval_plus": "code",
    "mbpp_plus": "code",
    "mnli": "nli",
    "snli": "nli",
}

ALL_DATASETS = list(DOMAIN_OF.keys())


def _sample(items: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    if n is None or n >= len(items):
        return items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:n]]


def load_gsm8k(n: int = 200) -> List[Dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = [{"id": f"gsm8k-{i}", "text": ex["question"].strip()}
             for i, ex in enumerate(ds)]
    return _sample(items, n)


def load_svamp(n: int = 200) -> List[Dict]:
    ds = load_dataset("ChilleD/SVAMP", split="test")
    items = []
    for i, ex in enumerate(ds):
        body = (ex.get("Body") or "").strip()
        q = (ex.get("Question") or "").strip()
        text = (body + " " + q).strip()
        items.append({"id": f"svamp-{i}", "text": text})
    return _sample(items, n)


def load_humaneval_plus(n: int = 200) -> List[Dict]:
    ds = load_dataset("evalplus/humanevalplus", split="test")
    items = [{"id": ex["task_id"], "text": ex["prompt"].rstrip()}
             for ex in ds]
    return _sample(items, n)


def load_mbpp_plus(n: int = 200) -> List[Dict]:
    ds = load_dataset("evalplus/mbppplus", split="test")
    items = [{"id": str(ex["task_id"]), "text": ex["prompt"].strip()}
             for ex in ds]
    return _sample(items, n)


def _nli_text(premise: str, hypothesis: str) -> str:
    return f"Premise: {premise.strip()} Hypothesis: {hypothesis.strip()}"


def load_mnli(n: int = 200) -> List[Dict]:
    ds = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
    items = []
    for i, ex in enumerate(ds):
        if ex["label"] < 0:
            continue
        items.append({"id": f"mnli-{i}",
                      "text": _nli_text(ex["premise"], ex["hypothesis"])})
    return _sample(items, n)


def load_snli(n: int = 200) -> List[Dict]:
    ds = load_dataset("stanfordnlp/snli", split="test")
    items = []
    for i, ex in enumerate(ds):
        if ex["label"] < 0:        # -1 == no gold label
            continue
        items.append({"id": f"snli-{i}",
                      "text": _nli_text(ex["premise"], ex["hypothesis"])})
    return _sample(items, n)


_LOADERS = {
    "gsm8k": load_gsm8k,
    "svamp": load_svamp,
    "humaneval_plus": load_humaneval_plus,
    "mbpp_plus": load_mbpp_plus,
    "mnli": load_mnli,
    "snli": load_snli,
}


def load(name: str, n: int = 200) -> List[Dict]:
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset {name}. Options: {ALL_DATASETS}")
    return _LOADERS[name](n)

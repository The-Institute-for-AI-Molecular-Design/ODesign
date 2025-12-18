# -*- coding: utf-8 -*-
import os
import json
from typing import List, Tuple, Optional, Union, Dict, Any, Iterable

import torch
import numpy as np
from torcheval.metrics.text import Perplexity
from torcheval.metrics.classification import MulticlassF1Score


TYPE_BUCKET_SPECS: Dict[str, List[Tuple[int, int, str]]] = {
    "protein": [(0, 100, "len_0_100"),
                (100, 300, "len_100_300"),
                (300, 500, "len_300_500"),
                (-1, 10**9, "len_all")],

    "rna":     [(0, 50,  "len_0_50"),
                (50, 100, "len_50_100"),
                (100, 500,"len_100_500"),
                (-1, 10**9, "len_all")],
    "dna":     [(0, 50,  "len_0_50"),
                (50, 100, "len_50_100"),
                (100, 500,"len_100_500"),
                (-1, 10**9, "len_all")],
}


def normalize_type(t) -> str:
    if t is None:
        return "protein"
    s = str(t).strip().lower()
    if s in ("protein", "prot", "p"): return "protein"
    if s in ("rna",): return "rna"
    if s in ("dna",): return "dna"
    return "protein"


def _as_spans_list(
    spans: Union[None, torch.Tensor, np.ndarray, List[List[int]], List[Tuple[int, int]]]
) -> List[Tuple[int, int]]:
    if spans is None:
        return []
    if isinstance(spans, list) and len(spans) == 1 and isinstance(spans[0], torch.Tensor):
        spans = spans[0]
    if isinstance(spans, torch.Tensor):
        arr = spans.detach().cpu().long().numpy()
    elif isinstance(spans, np.ndarray):
        arr = spans
    else:
        try:
            arr = np.asarray(spans)
        except Exception:
            return []
    if arr.ndim != 2 or arr.shape[1] != 2:
        return []
    out: List[Tuple[int, int]] = []
    for s, e in arr.tolist():
        s = int(s); e = int(e)
        if e > s >= 0:
            out.append((s, e))
    return out


def iter_chain_segments(
    log_probs: torch.Tensor,     # [N,V] or [B,N,V]
    target: torch.Tensor,        # [N] or [B,N]
    mask: torch.Tensor,          # [N] / [B,N]
    batch: Dict[str, Any],
):
    """
    Yield per-chain segments:
        (logits2d[T,V], tgt1d[T], L=T, tname:str, sample_id:int).

    Priority:
        1. Use chain_spans for segmentation if available.
        2. Chain type resolved from batch['type'] per chain; fallback to majority voting in type_vec.
        3. If no spans, treat sample as a single segment.
    """

    tv = batch.get('type_vec', None)
    bid = batch.get('batch_id', None)
    chain_types_all = batch.get('type', None)

    def _majority_type_from_slice(tv_1d: Optional[torch.Tensor], sel_1d: torch.Tensor) -> str:
        if tv_1d is None or (not torch.any(sel_1d)):
            return "protein"
        vals = tv_1d[sel_1d].detach().cpu().tolist()
        if not vals:
            return "protein"
        maj = max((0, 1, 2), key=lambda x: vals.count(x))
        return {0: "protein", 1: "rna", 2: "dna"}[maj]

    if log_probs.dim() == 3:
        B, N, V = log_probs.shape
        spans_all = batch.get('chain_spans', None)

        spans_per_sample: List[List[Tuple[int,int]]] = [[] for _ in range(B)]
        if spans_all is not None:
            if isinstance(spans_all, list) and len(spans_all) == B:
                for i in range(B):
                    spans_per_sample[i] = _as_spans_list(spans_all[i])
            elif isinstance(spans_all, torch.Tensor) and spans_all.dim() == 3 and spans_all.shape[0] == B:
                for i in range(B):
                    spans_per_sample[i] = _as_spans_list(spans_all[i])
            elif isinstance(spans_all, np.ndarray) and spans_all.ndim == 3 and spans_all.shape[0] == B:
                for i in range(B):
                    spans_per_sample[i] = _as_spans_list(spans_all[i])

        for i in range(B):
            mi = (mask[i] > 0)
            if not torch.any(mi):
                continue
            spans_i = spans_per_sample[i]
            sample_id = int(i)

            if isinstance(chain_types_all, list) and len(chain_types_all) == B and isinstance(chain_types_all[i], (list, tuple)):
                chain_types_i = list(chain_types_all[i])
            else:
                chain_types_i = None

            if spans_i:
                for j, (s, e) in enumerate(spans_i):
                    sel = mi[s:e]
                    if not torch.any(sel):
                        continue
                    logits2d = log_probs[i, s:e][sel]
                    tgt1d    = target[i, s:e][sel]
                    if chain_types_i is not None and j < len(chain_types_i):
                        tname = normalize_type(chain_types_i[j])
                    else:
                        tv2 = tv[i] if tv is not None and tv.dim() == 2 else None
                        tname = _majority_type_from_slice(tv2, sel)
                    yield (logits2d, tgt1d, int(sel.sum().item()), tname, sample_id)
            else:
                logits2d = log_probs[i][mi]
                tgt1d    = target[i][mi]
                tv2 = tv[i] if tv is not None and tv.dim() == 2 else None
                tname = _majority_type_from_slice(tv2, mi)
                yield (logits2d, tgt1d, int(mi.sum().item()), tname, sample_id)
        return

    N, V = log_probs.shape
    m_all = (mask > 0)
    spans_all = batch.get('chain_spans', None)

    if isinstance(bid, torch.Tensor) and bid.numel() == N:
        uniq = torch.unique(bid).sort()[0].tolist()

        spans_list = None
        if isinstance(spans_all, list) and len(spans_all) == len(uniq):
            spans_list = [ _as_spans_list(x) for x in spans_all ]
        elif isinstance(spans_all, torch.Tensor) and spans_all.dim() == 3 and spans_all.shape[0] == len(uniq):
            spans_list = [ _as_spans_list(spans_all[i]) for i in range(len(uniq)) ]
        elif isinstance(spans_all, np.ndarray) and spans_all.ndim == 3 and spans_all.shape[0] == len(uniq):
            spans_list = [ _as_spans_list(spans_all[i]) for i in range(len(uniq)) ]

        for idx, b in enumerate(uniq):
            sel_b = (bid == b) & m_all
            if not torch.any(sel_b):
                continue
            sample_id = int(b)

            chain_types_i = None
            if isinstance(chain_types_all, list) and len(chain_types_all) == len(uniq) and isinstance(chain_types_all[idx], (list, tuple)):
                chain_types_i = list(chain_types_all[idx])

            if spans_list and len(spans_list) > idx and spans_list[idx]:
                idxs = torch.nonzero(bid == b, as_tuple=False).view(-1)
                g0 = idxs[0].item()
                for j, (s, e) in enumerate(spans_list[idx]):
                    rng = torch.arange(g0 + s, g0 + e, device=log_probs.device)
                    sel = (m_all.index_select(0, rng) > 0)
                    if not torch.any(sel):
                        continue
                    logits2d = log_probs.index_select(0, rng)[sel]
                    tgt1d    = target.index_select(0, rng)[sel]
                    if chain_types_i is not None and j < len(chain_types_i):
                        tname = normalize_type(chain_types_i[j])
                    else:
                        tv_seg = (tv.index_select(0, rng) if tv is not None and tv.dim() == 1 else None)
                        tname  = _majority_type_from_slice(tv_seg, sel)
                    yield (logits2d, tgt1d, int(sel.sum().item()), tname, sample_id)
            else:
                logits2d = log_probs[sel_b]
                tgt1d    = target[sel_b]
                tv_b     = (tv[sel_b] if tv is not None and tv.dim() == 1 else None)
                if isinstance(chain_types_all, list) and len(chain_types_all) == len(uniq) and isinstance(chain_types_all[idx], str):
                    tname = normalize_type(chain_types_all[idx])
                else:
                    tname = _majority_type_from_slice(tv_b, torch.ones_like(tgt1d, dtype=torch.bool))
                yield (logits2d, tgt1d, int(sel_b.sum().item()), tname, sample_id)
        return

    if spans_all is not None:
        spans = _as_spans_list(spans_all)
        if spans:
            chain_types_i = list(chain_types_all) if isinstance(chain_types_all, (list, tuple)) else None
            sample_id = 0
            for j, (s, e) in enumerate(spans):
                sel = m_all[s:e]
                if not torch.any(sel):
                    continue
                logits2d = log_probs[s:e][sel]
                tgt1d    = target[s:e][sel]
                if chain_types_i is not None and j < len(chain_types_i):
                    tname = normalize_type(chain_types_i[j])
                else:
                    tv_seg = (tv[s:e] if tv is not None and tv.dim() == 1 else None)
                    tname  = _majority_type_from_slice(tv_seg, sel)
                yield (logits2d, tgt1d, int(sel.sum().item()), tname, sample_id)
            return

    if torch.any(m_all):
        logits2d = log_probs[m_all]
        tgt1d    = target[m_all]
        if isinstance(chain_types_all, str):
            tname = normalize_type(chain_types_all)
        else:
            tname = _majority_type_from_slice(tv if tv is not None and tv.dim() == 1 else None,
                                             torch.ones_like(tgt1d, dtype=torch.bool))
        sample_id = 0
        yield (logits2d, tgt1d, int(m_all.sum().item()), tname, sample_id)


class EvalAccumulator:
    """
    Evaluation accumulator supporting:
        - Overall metrics (perplexity, macro F1, recovery).
        - Per-type and per-length-bucket metrics.
        - Recovery is computed as per-sample average of segment-level accuracies.
    """
    def __init__(self, vocab_size: int, quiet_f1_warning: bool = True):
        self.vocab_size = vocab_size

        self.overall_ppl = Perplexity()
        self.overall_f1  = MulticlassF1Score(average='macro',
                                             num_classes=vocab_size)

        self.type_bucket_specs = TYPE_BUCKET_SPECS

        self.bucket_ppl: Dict[Tuple[str, str], Perplexity] = {}
        self.bucket_f1m: Dict[Tuple[str, str], MulticlassF1Score] = {}
        self.bucket_rec_num: Dict[Tuple[str, str], int] = {}
        self.bucket_rec_den: D

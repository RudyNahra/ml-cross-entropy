from __future__ import annotations
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from typing import TypeVar

import torch
import transformers

from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.cce_utils import CCEPreset

TransformersModelT = TypeVar("TransformersModelT", bound=transformers.PreTrainedModel)


class CCEKwargs(CCEPreset):
    impl: str
    reduction: str


@dataclass
class PatchOptions:
    impl: str
    reduction: str
    filter_eps: float | str | None
    accum_e_fp32: bool
    accum_c_fp32: bool
    filter_e_grad: bool
    filter_c_grad: bool
    train_only: bool

    def to_kwargs(self) -> CCEKwargs:
        return CCEKwargs(
            impl=self.impl,
            reduction=self.reduction,
            filter_eps=self.filter_eps,
            accum_e_fp32=self.accum_e_fp32,
            accum_c_fp32=self.accum_c_fp32,
            filter_e_grad=self.filter_e_grad,
            filter_c_grad=self.filter_c_grad,
        )

    def use_lce(self, labels: torch.Tensor | None, training: bool) -> bool:
        if labels is None:
            return False

        if not training and self.train_only:
            return False

        return True


def apply_lce(
    e: torch.Tensor,
    c: torch.Tensor,
    labels: torch.Tensor,
    opts: PatchOptions,
    bias: torch.Tensor | None = None,
    **loss_kwargs,
) -> torch.Tensor:
    num_items_in_batch = loss_kwargs.get("num_items_in_batch", None)
    loss_weights = loss_kwargs.get("loss_weights", None)
    
    cce_kwargs = opts.to_kwargs()
    
    # If per-token weights are provided, use reduction='none' and apply weights
    if loss_weights is not None:
        cce_kwargs["reduction"] = "none"
        per_token_loss = linear_cross_entropy(
            e,
            c,
            labels.to(e.device),
            bias=bias,
            shift=True,
            **cce_kwargs,
        )
        # per_token_loss is [batch, seq_len-1] due to shift=True
        # loss_weights should be [batch, seq_len], so shift it to [batch, seq_len-1]
        shifted_weights = loss_weights[:, 1:]
        weighted_loss = per_token_loss * shifted_weights
        total_weight = shifted_weights.sum()
        loss = weighted_loss.sum() / total_weight if total_weight > 0 else torch.tensor(0.0, device=e.device)
    else:
        # Standard path: use the configured reduction
        if num_items_in_batch is not None and cce_kwargs["reduction"] == "mean":
            cce_kwargs["reduction"] = "sum"
        else:
            num_items_in_batch = None

        loss = linear_cross_entropy(
            e,
            c,
            labels.to(e.device),
            bias=bias,
            shift=True,
            **cce_kwargs,
        )

        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch

    return loss

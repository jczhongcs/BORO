from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve


class BOROLoss(nn.Module):
    """BORO loss for precision-constrained binary scoring.

    A precision operating boundary proxy is estimated from cross-epoch score
    history, positives are weighted by their boundary utility, and training
    optimizes positive/negative ranking around that boundary. It intentionally
    contains no BCE/CE auxiliary loss.
    """

    def __init__(
        self,
        target_precision: float = 0.95,
        initial_margin: float = 0.2,
        positive_pair_weight: float = 1.0,
        negative_pair_weight: float = 1.0,
        history_pair_weight: float = 1.0,
        current_pair_weight: float = 1.0,
        scale_mode: str = "boundary_band",
        upper_temperature: float = 1.0,
        lower_temperature: float = 1.0,
        lower_band: float = 0.5,
        min_pos_weight: float = 0.35,
        far_below_suppression: float = 0.10,
        boundary_start_epoch: int = 3,
        threshold_momentum: float = 0.50,
        max_prev_positives: int = 5000,
        max_prev_negatives: int = 5000,
        use_negative_scale: bool = False,
        negative_temperature: float = 1.0,
        min_negative_weight: float = 0.05,
        current_hard_negatives: int = 64,
        soft_rank_temperature: float = 1.0,
        boundary_negative_weight: float = 1.0,
        boundary_gap_weight: float = 1.0,
    ):
        super().__init__()
        min_pos_weight = float(min_pos_weight)
        if not (0.0 < min_pos_weight <= 1.0):
            raise ValueError("min_pos_weight must satisfy 0 < min_pos_weight <= 1 for BORO.")
        self.target_precision = float(target_precision)
        self.positive_pair_weight = float(positive_pair_weight)
        self.negative_pair_weight = float(negative_pair_weight)
        self.history_pair_weight = float(history_pair_weight)
        self.current_pair_weight = float(current_pair_weight)
        self.scale_mode = str(scale_mode)
        self.upper_temperature = max(float(upper_temperature), 1e-6)
        self.lower_temperature = max(float(lower_temperature), 1e-6)
        self.lower_band = float(lower_band)
        self.min_pos_weight = min_pos_weight
        self.far_below_suppression = float(np.clip(far_below_suppression, 0.0, 0.95))
        self.boundary_start_epoch = int(max(boundary_start_epoch, 0))
        self.threshold_momentum = float(threshold_momentum)
        self.max_prev_positives = int(max_prev_positives)
        self.max_prev_negatives = int(max_prev_negatives)
        self.use_negative_scale = bool(use_negative_scale)
        self.negative_temperature = max(float(negative_temperature), 1e-6)
        self.min_negative_weight = float(np.clip(min_negative_weight, 0.0, 1.0))
        self.current_hard_negatives = int(max(current_hard_negatives, 0))
        self.soft_rank_temperature = max(float(soft_rank_temperature), 1e-6)
        self.boundary_negative_weight = float(max(boundary_negative_weight, 0.0))
        self.boundary_gap_weight = float(max(boundary_gap_weight, 0.0))

        self.margin = nn.Parameter(torch.tensor(float(initial_margin), dtype=torch.float32))
        self.prev_scores: torch.Tensor | None = None
        self.prev_labels: torch.Tensor | None = None
        self.raw_tau_p: float | None = None
        self.tau_p: float | None = None
        self.history_update_count = 0

    def update_history(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        self.prev_scores = scores.detach().float().cpu()
        self.prev_labels = labels.detach().float().cpu()
        raw_tau = self._compute_precision_threshold(self.prev_scores, self.prev_labels)
        self.raw_tau_p = raw_tau
        if self.tau_p is None or self.threshold_momentum <= 0.0:
            self.tau_p = raw_tau
        else:
            self.tau_p = self.threshold_momentum * self.tau_p + (1.0 - self.threshold_momentum) * raw_tau
        self.history_update_count += 1
        return float(self.tau_p)

    def clamp_margin_(self) -> None:
        with torch.no_grad():
            self.margin.clamp_(min=0.0)

    def initialize_history(self, labels: torch.Tensor, seed: int = 0) -> None:
        labels = labels.detach().float().view(-1).cpu()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        self.prev_scores = torch.rand(labels.numel(), generator=generator).float().cpu() * 0.5
        self.prev_labels = labels
        self.raw_tau_p = None
        self.tau_p = None
        self.history_update_count = 0

    def _compute_precision_threshold(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        y = labels.detach().cpu().numpy().astype(int)
        s = scores.detach().cpu().numpy().astype(float)
        precision, recall, thresholds = precision_recall_curve(y, s)
        valid = np.flatnonzero(precision[:-1] >= self.target_precision)
        if valid.size > 0:
            best = valid[np.argmax(recall[:-1][valid])]
            return float(thresholds[best])
        positives = s[y == 1]
        if positives.size > 0:
            return float(np.max(positives))
        return float(np.max(s))

    def _adaptive_scale(self, current_pos: torch.Tensor) -> torch.Tensor:
        if current_pos.numel() == 0 or self.tau_p is None:
            return torch.ones_like(current_pos)
        tau = torch.tensor(float(self.tau_p), device=current_pos.device, dtype=current_pos.dtype)
        scale_scores = current_pos.detach()

        if self.scale_mode == "operating_exp" or self.history_update_count < self.boundary_start_epoch:
            excess = torch.clamp(scale_scores - tau, min=0.0)
            return torch.exp(-excess / self.upper_temperature)

        if self.scale_mode != "boundary_band":
            raise ValueError(f"Unsupported scale_mode: {self.scale_mode}")

        above_decay = torch.exp(-torch.clamp(scale_scores - tau, min=0.0) / self.upper_temperature)
        far_below_gate = torch.sigmoid((tau - scale_scores - self.lower_band) / self.lower_temperature)
        below_factor = 1.0 - self.far_below_suppression * far_below_gate
        scale = below_factor * above_decay
        return torch.clamp(scale, min=self.min_pos_weight, max=1.0)

    def _negative_scale(self, current_neg: torch.Tensor) -> torch.Tensor:
        if (not self.use_negative_scale) or current_neg.numel() == 0 or self.tau_p is None:
            return torch.ones_like(current_neg)
        tau = torch.tensor(float(self.tau_p), device=current_neg.device, dtype=current_neg.dtype)
        gate = torch.sigmoid((current_neg.detach() - tau) / self.negative_temperature)
        scale = self.min_negative_weight + (1.0 - self.min_negative_weight) * gate
        return torch.clamp(scale, min=self.min_negative_weight, max=1.0)

    def _sample_history(self, scores: torch.Tensor, cap: int) -> torch.Tensor:
        if cap <= 0 or scores.numel() <= cap:
            return scores
        idx = torch.randperm(scores.numel(), device=scores.device)[:cap]
        return scores[idx]

    def _select_current_negatives(self, current_neg: torch.Tensor) -> torch.Tensor:
        if current_neg.numel() == 0:
            return current_neg
        if self.current_hard_negatives <= 0 or current_neg.numel() <= self.current_hard_negatives:
            return current_neg
        return torch.topk(current_neg, k=self.current_hard_negatives).values

    def _boundary_value(self, reference_neg: torch.Tensor, fallback_scores: torch.Tensor) -> torch.Tensor:
        if self.tau_p is not None:
            return torch.tensor(float(self.tau_p), device=fallback_scores.device, dtype=fallback_scores.dtype)
        if reference_neg.numel() > 0:
            return torch.quantile(reference_neg.detach().float().to(fallback_scores.device), q=0.95).to(fallback_scores.dtype)
        return fallback_scores.detach().mean()

    def _current_boundary_rank_loss(self, current_pos: torch.Tensor, current_neg: torch.Tensor, prev_neg: torch.Tensor) -> torch.Tensor:
        if current_pos.numel() == 0 or current_neg.numel() == 0:
            return current_pos.sum() * 1e-9 if current_pos.numel() else current_neg.sum() * 1e-9

        selected_neg = self._select_current_negatives(current_neg)
        margin = torch.clamp(self.margin, min=0.0)
        temperature = torch.tensor(self.soft_rank_temperature, device=current_pos.device, dtype=current_pos.dtype)
        boundary = self._boundary_value(prev_neg, current_neg)
        scale = self._adaptive_scale(current_pos)

        band_width = max(abs(self.lower_band), 1e-6)
        with torch.no_grad():
            band_gate = torch.exp(-torch.abs(selected_neg.detach() - boundary) / band_width)
            violation_gate = torch.sigmoid((selected_neg.detach().view(1, -1) - current_pos.detach().view(-1, 1) + margin.detach()) / self.lower_temperature)
            neg_weight = 1.0 + self.boundary_negative_weight * band_gate.view(1, -1) * violation_gate
            neg_log_weight = torch.log(torch.clamp(neg_weight, min=1e-12))

        pos_logits = current_pos.view(-1, 1)
        neg_logits = selected_neg.view(1, -1) + margin + temperature * neg_log_weight
        logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
        listwise = -torch.log_softmax(logits, dim=1)[:, 0]
        boundary_gap = temperature * torch.nn.functional.softplus((boundary + margin - current_pos) / temperature)
        return (scale * (listwise + self.boundary_gap_weight * boundary_gap)).mean()

    def forward(self, current_scores: torch.Tensor, current_labels: torch.Tensor) -> torch.Tensor:
        device = current_scores.device
        current_labels = current_labels.float()

        current_is_pos = current_labels >= 0.5
        current_pos = current_scores[current_is_pos]
        current_neg = current_scores[~current_is_pos]

        history_components = []
        margin = torch.clamp(self.margin, min=0.0)
        prev_neg = current_neg.detach()

        if self.prev_scores is not None and self.prev_labels is not None:
            prev_scores = self.prev_scores.to(device)
            prev_labels = self.prev_labels.to(device)
            prev_is_pos = prev_labels >= 0.5
            prev_pos = self._sample_history(prev_scores[prev_is_pos], self.max_prev_positives)
            prev_neg = self._sample_history(prev_scores[~prev_is_pos], self.max_prev_negatives)

            if current_pos.numel() > 0 and prev_neg.numel() > 0:
                diff = prev_neg.view(1, -1) - current_pos.view(-1, 1) + margin
                hinge_sq = torch.clamp(diff, min=0.0).pow(2)
                scale = self._adaptive_scale(current_pos).view(-1, 1)
                history_components.append(self.positive_pair_weight * (hinge_sq * scale).mean())

            if current_neg.numel() > 0 and prev_pos.numel() > 0:
                diff = current_neg.view(-1, 1) - prev_pos.view(1, -1) + margin
                hinge_sq = torch.clamp(diff, min=0.0).pow(2)
                if self.use_negative_scale:
                    hinge_sq = hinge_sq * self._negative_scale(current_neg).view(-1, 1)
                history_components.append(self.negative_pair_weight * hinge_sq.mean())

        history_loss = torch.stack(history_components).sum() if history_components else current_scores.sum() * 1e-9
        current_loss = self._current_boundary_rank_loss(current_pos, current_neg, prev_neg)
        loss = self.history_pair_weight * history_loss + self.current_pair_weight * current_loss

        return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)

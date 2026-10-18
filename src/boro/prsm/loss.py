import torch
import math
from numbers import Integral
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Optional, Union
import numpy as np

class AdaptiveRankLoss(nn.Module):

    def __init__(self, max_prev_targets: int=5000, max_prev_decoys: int=5000, constraint_value: float=0.01, lower_band: float=0.26, min_pos_weight: float=0.35, upper_temperature: float=1.0, lower_temperature: float=0.1, far_below_suppression: float=0.1, boundary_start_epoch: int=3, threshold_momentum: float=0.5, fallback_quantile: float=0.99, exact_scan_max_items: int=200000, num_bins: int=4096, min_required_targets: int=1, eps: float=1e-12, verbose: bool=False, history_pair_weight: float=1.0, current_pair_weight: float=0.003, current_hard_negatives: int=4, soft_rank_temperature: float=0.6, violation_temperature: float=0.1, boundary_negative_weight: float=0.5, boundary_gap_weight: float=0.15):
        super().__init__()
        min_pos_weight = float(min_pos_weight)
        if not 0.0 < min_pos_weight <= 1.0:
            raise ValueError('min_pos_weight must satisfy 0 < min_pos_weight <= 1.')
        self.max_prev_targets = int(max_prev_targets)
        self.max_prev_decoys = int(max_prev_decoys)
        self.constraint_value = float(constraint_value)
        self.lower_band = float(lower_band)
        self.min_pos_weight = min_pos_weight
        self.upper_temperature = float(max(upper_temperature, 1e-06))
        self.lower_temperature = float(max(lower_temperature, 1e-06))
        self.far_below_suppression = float(min(max(far_below_suppression, 0.0), 0.95))
        self.boundary_start_epoch = int(max(boundary_start_epoch, 0))
        self.threshold_momentum = float(threshold_momentum)
        self.fallback_quantile = float(min(max(fallback_quantile, 0.0), 1.0))
        self.exact_scan_max_items = int(exact_scan_max_items)
        self.num_bins = int(num_bins)
        self.min_required_targets = int(min_required_targets)
        self.eps = eps
        self.verbose = verbose
        self.prev_scores: Optional[torch.Tensor] = None
        self.prev_labels: Optional[torch.Tensor] = None
        self.cached_tau: Optional[torch.Tensor] = None
        self.raw_tau: Optional[torch.Tensor] = None
        self.threshold_history: List[float] = []
        self.history_update_count = 0
        for key, value in {'history_pair_weight': history_pair_weight, 'current_pair_weight': current_pair_weight, 'boundary_negative_weight': boundary_negative_weight, 'boundary_gap_weight': boundary_gap_weight}.items():
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(key)
            setattr(self, key, float(value))
        for key, value in {'soft_rank_temperature': soft_rank_temperature, 'violation_temperature': violation_temperature}.items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(key)
            setattr(self, key, float(value))
        if isinstance(current_hard_negatives, bool) or not isinstance(current_hard_negatives, Integral):
            raise ValueError('K must be an integer')
        self.current_hard_negatives = int(current_hard_negatives)
        if not (math.isfinite(self.lower_band) and self.lower_band > 0):
            raise ValueError('ell must be positive')
        if not 0 <= self.threshold_momentum < 1:
            raise ValueError('mu must be in [0,1)')
        self.last_components = {}

    def initialize_history(self, initial_labels: Union[List[int], np.ndarray, torch.Tensor]):
        if isinstance(initial_labels, list):
            initial_labels = torch.tensor(initial_labels)
        elif isinstance(initial_labels, np.ndarray):
            initial_labels = torch.from_numpy(initial_labels)
        initial_labels = initial_labels.float().cpu()
        self.prev_scores = torch.rand(len(initial_labels)).float().cpu() * 0.5
        self.prev_labels = initial_labels
        print(f'AdaptiveRankLoss initialized with {len(initial_labels)} samples. constraint=dtr:{self.constraint_value}, scale=boundary_band, initial_tau=warmup')

    def update_history(self, epoch_scores: torch.Tensor, epoch_labels: torch.Tensor):
        self.prev_scores = epoch_scores.detach().float().cpu()
        self.prev_labels = epoch_labels.detach().float().cpu()
        self.history_update_count += 1
        self._update_cached_threshold(self.prev_scores, self.prev_labels)

    def get_cached_tau(self) -> float:
        if self.cached_tau is None:
            return float('nan')
        return float(self.cached_tau.detach().cpu().item())

    def _update_cached_threshold(self, scores: torch.Tensor, labels: torch.Tensor):
        with torch.no_grad():
            tau = self._estimate_operating_threshold(scores, labels).detach().cpu()
            self.raw_tau = tau
            if self.cached_tau is None or self.threshold_momentum <= 0:
                self.cached_tau = tau
            else:
                m = self.threshold_momentum
                self.cached_tau = (m * self.cached_tau.detach().cpu() + (1.0 - m) * tau).detach().cpu()
            self.threshold_history.append(float(self.cached_tau.item()))
            if self.verbose:
                print(f'[AdaptiveRankLoss] raw_tau={float(tau):.6f}, cached_tau={self.get_cached_tau():.6f}')

    def _estimate_operating_threshold(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scores = scores.detach().float().view(-1).cpu()
        labels = labels.detach().float().view(-1).cpu()
        if scores.numel() == 0:
            return torch.tensor(0.0)
        if scores.numel() <= self.exact_scan_max_items:
            return self._estimate_operating_threshold_exact(scores, labels)
        return self._estimate_operating_threshold_histogram(scores, labels)

    def _estimate_operating_threshold_exact(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(scores, descending=True)
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        tp = torch.cumsum((sorted_labels >= 0.5).float(), dim=0)
        fp = torch.cumsum((sorted_labels < 0.5).float(), dim=0)
        feasible = (fp / (tp + self.eps) <= self.constraint_value) & (tp >= self.min_required_targets)
        if torch.any(feasible):
            k = torch.nonzero(feasible, as_tuple=False)[-1].item()
            return sorted_scores[k]
        return self._fallback_threshold(scores, labels)

    def _estimate_operating_threshold_histogram(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        s_min = float(torch.min(scores).item())
        s_max = float(torch.max(scores).item())
        if not np.isfinite(s_min) or not np.isfinite(s_max) or abs(s_max - s_min) < 1e-12:
            return self._fallback_threshold(scores, labels)
        pos_scores = scores[labels >= 0.5]
        neg_scores = scores[labels < 0.5]
        hist_pos = torch.histc(pos_scores, bins=self.num_bins, min=s_min, max=s_max) if pos_scores.numel() > 0 else torch.zeros(self.num_bins)
        hist_neg = torch.histc(neg_scores, bins=self.num_bins, min=s_min, max=s_max) if neg_scores.numel() > 0 else torch.zeros(self.num_bins)
        tp = torch.cumsum(torch.flip(hist_pos, dims=[0]), dim=0)
        fp = torch.cumsum(torch.flip(hist_neg, dims=[0]), dim=0)
        feasible = (fp / (tp + self.eps) <= self.constraint_value) & (tp >= self.min_required_targets)
        if torch.any(feasible):
            idx_desc = torch.nonzero(feasible, as_tuple=False)[-1].item()
            idx_low_to_high = self.num_bins - 1 - idx_desc
            bin_width = (s_max - s_min) / self.num_bins
            return torch.tensor(float(s_min + idx_low_to_high * bin_width))
        return self._fallback_threshold(scores, labels)

    def _fallback_threshold(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        decoys = scores[labels < 0.5]
        if decoys.numel() == 0:
            return torch.max(scores)
        if decoys.numel() == 1:
            return decoys[0]
        return torch.quantile(decoys.float(), q=self.fallback_quantile)

    def _compute_adaptive_scale(self, current_targets: torch.Tensor, prev_decoys: torch.Tensor) -> torch.Tensor:
        if current_targets.numel() == 0:
            return torch.ones_like(current_targets)
        scale_scores = current_targets.detach()
        if self.cached_tau is None:
            tau = torch.quantile(prev_decoys.detach().float().to(current_targets.device), q=self.fallback_quantile) if prev_decoys.numel() > 0 else torch.max(scale_scores.detach())
        else:
            tau = self.cached_tau.detach().to(current_targets.device)
        if self.history_update_count < self.boundary_start_epoch:
            return torch.exp(-torch.clamp(scale_scores - tau, min=0.0))
        above_decay = torch.exp(-torch.clamp((scale_scores - tau) / self.upper_temperature, min=0.0))
        far_below_gate = torch.sigmoid((tau - scale_scores - self.lower_band) / self.lower_temperature)
        below_factor = 1.0 - self.far_below_suppression * far_below_gate
        scale = below_factor * above_decay
        return torch.clamp(scale, min=self.min_pos_weight, max=1.0)

    def _subsample_history(self, prev_targets: torch.Tensor, prev_decoys: torch.Tensor, device: torch.device):
        target_prob = min(1.0, self.max_prev_targets / max(1, len(prev_targets)))
        decoy_prob = min(1.0, self.max_prev_decoys / max(1, len(prev_decoys)))
        sampled_targets = prev_targets[torch.rand(len(prev_targets), device=device) < target_prob]
        sampled_decoys = prev_decoys[torch.rand(len(prev_decoys), device=device) < decoy_prob]
        return (sampled_targets, sampled_decoys)

    @staticmethod
    def _cap_normalizer(cap: int) -> float:
        return float(max(1, cap))

    def _historical_forward(self, current_scores: torch.Tensor, current_labels: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
        if self.prev_scores is None or self.prev_labels is None:
            return torch.tensor(0.0, device=current_scores.device, requires_grad=True)
        device = current_scores.device
        prev_scores = self.prev_scores.to(device)
        prev_labels = self.prev_labels.to(device)
        current_is_target = current_labels >= 0.5
        prev_is_target = prev_labels >= 0.5
        num_current_targets = torch.sum(current_is_target)
        if num_current_targets == 0 or num_current_targets == len(current_is_target):
            return torch.sum(current_scores) * 1e-09
        current_targets = current_scores[current_is_target]
        current_decoys = current_scores[~current_is_target]
        prev_targets = prev_scores[prev_is_target]
        prev_decoys = prev_scores[~prev_is_target]
        adaptive_scale = self._compute_adaptive_scale(current_targets, prev_decoys)
        sampled_prev_targets, sampled_prev_decoys = self._subsample_history(prev_targets, prev_decoys, device)
        history_loss_components = []
        if len(current_targets) > 0 and len(sampled_prev_decoys) > 0:
            diff_matrix = sampled_prev_decoys.view(1, -1) - current_targets.view(-1, 1) + margin
            squared_loss = torch.clamp(diff_matrix, min=0) ** 2
            scaled_loss = squared_loss * adaptive_scale.view(-1, 1)
            history_loss_components.append(torch.sum(scaled_loss) / self._cap_normalizer(self.max_prev_targets))
        if len(current_decoys) > 0 and len(sampled_prev_targets) > 0:
            diff_matrix = current_decoys.view(-1, 1) - sampled_prev_targets.view(1, -1) + margin
            squared_loss = torch.clamp(diff_matrix, min=0) ** 2
            history_loss_components.append(torch.sum(squared_loss) / self._cap_normalizer(self.max_prev_decoys))
        history_loss = sum(history_loss_components) if history_loss_components else torch.tensor(0.0, device=device)
        total_loss = history_loss
        final_loss = torch.where(torch.isnan(total_loss), torch.zeros_like(total_loss), total_loss)
        if not final_loss.requires_grad and (current_scores.requires_grad or margin.requires_grad):
            final_loss = final_loss + 0.0 * (current_scores.sum() + margin)
        return final_loss

    def current_components(self, scores, labels, margin):
        positive = scores[labels >= 0.5]
        negative = scores[labels < 0.5]
        zero = scores.sum() * 0 + margin * 0
        if positive.numel() == 0 or negative.numel() == 0:
            return (zero, zero)
        old_negative = negative.detach()
        if self.prev_scores is not None and self.prev_labels is not None:
            old_negative = self.prev_scores.detach()[self.prev_labels.detach() < 0.5].to(scores)
        boundary = self.cached_tau.detach().to(scores) if self.cached_tau is not None else torch.quantile(old_negative.detach(), self.fallback_quantile) if old_negative.numel() else positive.detach().max()
        scale = self._compute_adaptive_scale(positive, old_negative).detach()
        if self.current_hard_negatives > 0 and negative.numel() > self.current_hard_negatives:
            indices = torch.argsort(negative.detach(), descending=True, stable=True)[:self.current_hard_negatives]
            negative = negative[indices]
        with torch.no_grad():
            proximity = torch.exp(-torch.abs(negative.detach() - boundary) / self.lower_band)
            violation = torch.sigmoid((negative.detach()[None, :] - positive.detach()[:, None] + margin.detach()) / self.violation_temperature)
            weight = 1 + self.boundary_negative_weight * proximity[None, :] * violation
        T = self.soft_rank_temperature
        logits = (negative[None, :] - positive[:, None] + margin) / T + torch.log(weight)
        listwise = torch.logsumexp(torch.cat([torch.zeros_like(positive[:, None]), logits], dim=1), dim=1)
        gap = T * F.softplus((boundary - positive + margin) / T)
        return ((scale * listwise).mean(), (scale * gap).mean())

    def forward(self, current_scores, current_labels, margin):
        scores, labels = (current_scores, current_labels)
        if self.prev_scores is not None:
            self.prev_scores = self.prev_scores.detach()
        if self.prev_labels is not None:
            self.prev_labels = self.prev_labels.detach()
        historical = self._historical_forward(scores, labels, margin)
        if self.current_pair_weight == 0:
            self.last_components = {'history': historical.detach(), 'current_listwise': scores.new_zeros(()), 'current_gap': scores.new_zeros(()), 'current_weighted': scores.new_zeros(())}
            return historical if self.history_pair_weight == 1 else self.history_pair_weight * historical
        if not ((labels >= 0.5).any() and (labels < 0.5).any()):
            historical = scores.sum() * 0
        listwise, gap = self.current_components(scores, labels, margin)
        current = self.current_pair_weight * (listwise + self.boundary_gap_weight * gap)
        self.last_components = {'history': historical.detach(), 'current_listwise': listwise.detach(), 'current_gap': gap.detach(), 'current_weighted': current.detach()}
        return self.history_pair_weight * historical + current

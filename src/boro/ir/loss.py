from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveRankLossIR(nn.Module):

    def __init__(self, max_prev_targets: int=5000, max_prev_decoys: int=5000, constraint_value: float=0.01, lower_band: float=0.26, min_pos_weight: float=0.35, max_pos_weight: float=1.0, boundary_start_epoch: int=3, threshold_momentum: float=0.5, fallback_quantile: float=0.99, exact_scan_max_items: int=200000, num_bins: int=4096, min_required_targets: int=1, margin_init: float=0.2, include_inbatch_decoys: bool=True, decoy_mode: str='all', score_mode: str='raw', hard_decoy_topk: int=0, soft_rank_temperature: float=0.05, boundary_decoy_weight: float=1.0, history_pair_weight: float=1.0, current_pair_weight: float=1.0, deterministic_seed: int=42, margin_lr: float=1e-05, eps: float=1e-12, verbose: bool=False):
        super().__init__()
        min_pos_weight = float(min_pos_weight)
        max_pos_weight = float(max_pos_weight)
        if not 0.0 < min_pos_weight <= 1.0:
            raise ValueError('min_pos_weight must satisfy 0 < min_pos_weight <= 1.')
        if max_pos_weight < min_pos_weight:
            raise ValueError('max_pos_weight must be greater than or equal to min_pos_weight.')
        self.max_prev_targets = int(max_prev_targets)
        self.max_prev_decoys = int(max_prev_decoys)
        self.constraint_value = float(constraint_value)
        self.lower_band = float(lower_band)
        self.min_pos_weight = min_pos_weight
        self.max_pos_weight = max_pos_weight
        self.hard_positive_boost = 0.0
        self.upper_temperature = 1.0
        self.lower_temperature = 0.1
        self.far_below_suppression = 0.1
        self.boundary_start_epoch = int(max(boundary_start_epoch, 0))
        self.threshold_momentum = float(threshold_momentum)
        self.fallback_quantile = float(min(max(fallback_quantile, 0.0), 1.0))
        self.exact_scan_max_items = int(exact_scan_max_items)
        self.num_bins = int(num_bins)
        self.min_required_targets = int(min_required_targets)
        self.include_inbatch_decoys = bool(include_inbatch_decoys)
        if not self.include_inbatch_decoys and decoy_mode == 'all':
            decoy_mode = 'paired'
        if decoy_mode not in {'all', 'explicit_all', 'paired'}:
            raise ValueError('decoy_mode must be one of: all, explicit_all, paired')
        if score_mode not in {'raw', 'query_centered', 'positive_gap'}:
            raise ValueError('score_mode must be one of: raw, query_centered, positive_gap')
        self.decoy_mode = decoy_mode
        self.score_mode = score_mode
        self.hard_decoy_topk = int(max(hard_decoy_topk, 0))
        self.soft_rank_temperature = float(max(soft_rank_temperature, 1e-06))
        self.boundary_decoy_weight = float(max(boundary_decoy_weight, 0.0))
        self.history_pair_weight = float(history_pair_weight)
        self.current_pair_weight = float(current_pair_weight)
        self.deterministic_seed = int(deterministic_seed)
        self._cpu_rng = torch.Generator(device='cpu')
        self._cpu_rng.manual_seed(self.deterministic_seed)
        self.margin_lr = float(margin_lr)
        self.eps = eps
        self.verbose = verbose
        self.margin = nn.Parameter(torch.tensor(float(margin_init)))
        self.prev_scores: Optional[torch.Tensor] = None
        self.prev_labels: Optional[torch.Tensor] = None
        self.cached_tau: Optional[torch.Tensor] = None
        self.raw_tau: Optional[torch.Tensor] = None
        self.threshold_history: List[float] = []
        self.history_update_count = 0
        self._epoch_scores: List[torch.Tensor] = []
        self._epoch_labels: List[torch.Tensor] = []
        self.threshold_estimate_count = 0
        self.fallback_count = 0
        self.exact_threshold_count = 0
        self.histogram_threshold_count = 0
        self._epoch_active_comparison_sum = 0.0
        self._epoch_boundary_selection_sum = 0.0
        self._epoch_current_batches = 0

    def initialize_history(self, initial_labels: torch.Tensor) -> None:
        labels = initial_labels.detach().float().view(-1).cpu()
        self.prev_scores = torch.rand(len(labels), generator=self._cpu_rng).float().cpu() * 0.5
        self.prev_labels = labels

    def update_history(self, epoch_scores: torch.Tensor, epoch_labels: torch.Tensor) -> None:
        self.prev_scores = epoch_scores.detach().float().view(-1).cpu()
        self.prev_labels = epoch_labels.detach().float().view(-1).cpu()
        self.history_update_count += 1
        self._update_cached_threshold(self.prev_scores, self.prev_labels)

    def on_epoch_end(self) -> Dict[str, float]:
        if self._epoch_scores and self._epoch_labels:
            scores = torch.cat(self._epoch_scores)
            labels = torch.cat(self._epoch_labels)
            self.update_history(scores, labels)
        self._epoch_scores = []
        self._epoch_labels = []
        history_bytes = 0
        if self.prev_scores is not None:
            history_bytes += self.prev_scores.numel() * self.prev_scores.element_size()
        if self.prev_labels is not None:
            history_bytes += self.prev_labels.numel() * self.prev_labels.element_size()
        batch_count = max(1, self._epoch_current_batches)
        result = {'arl_tau': self.get_cached_tau(), 'arl_margin': float(torch.clamp(self.margin.detach().cpu(), min=0.0)), 'arl_history_updates': float(self.history_update_count), 'arl_threshold_estimates': float(self.threshold_estimate_count), 'arl_fallback_count': float(self.fallback_count), 'arl_fallback_rate': float(self.fallback_count / max(1, self.threshold_estimate_count)), 'arl_exact_threshold_count': float(self.exact_threshold_count), 'arl_histogram_threshold_count': float(self.histogram_threshold_count), 'arl_active_comparison_rate': float(self._epoch_active_comparison_sum / batch_count), 'arl_boundary_selection_rate': float(self._epoch_boundary_selection_sum / batch_count), 'arl_history_state_bytes': float(history_bytes)}
        self._epoch_active_comparison_sum = 0.0
        self._epoch_boundary_selection_sum = 0.0
        self._epoch_current_batches = 0
        return result

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            self.margin.clamp_(min=0.0)

    def get_cached_tau(self) -> float:
        if self.cached_tau is None:
            return float('nan')
        return float(self.cached_tau.detach().cpu().item())

    def _update_cached_threshold(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
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
                print(f'[AdaptiveRankLossIR] raw_tau={float(tau):.6f}, cached_tau={self.get_cached_tau():.6f}')

    def _estimate_operating_threshold(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.threshold_estimate_count += 1
        scores = scores.detach().float().view(-1).cpu()
        labels = labels.detach().float().view(-1).cpu()
        finite = torch.isfinite(scores)
        scores = scores[finite]
        labels = labels[finite]
        if scores.numel() == 0:
            return torch.tensor(0.0)
        if scores.numel() <= self.exact_scan_max_items:
            self.exact_threshold_count += 1
            return self._estimate_operating_threshold_exact(scores, labels)
        self.histogram_threshold_count += 1
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
        if abs(s_max - s_min) < 1e-12:
            return self._fallback_threshold(scores, labels)
        pos_scores = scores[labels >= 0.5]
        neg_scores = scores[labels < 0.5]
        hist_pos = torch.histc(pos_scores, bins=self.num_bins, min=s_min, max=s_max) if pos_scores.numel() else torch.zeros(self.num_bins)
        hist_neg = torch.histc(neg_scores, bins=self.num_bins, min=s_min, max=s_max) if neg_scores.numel() else torch.zeros(self.num_bins)
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
        self.fallback_count += 1
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
            scale = torch.exp(-torch.clamp(scale_scores - tau, min=0.0))
            if self.hard_positive_boost > 0:
                hard_gate = torch.sigmoid((tau + self.lower_band - scale_scores) / self.lower_temperature)
                scale = scale * (1.0 + self.hard_positive_boost * hard_gate)
            return torch.clamp(scale, min=self.min_pos_weight, max=self.max_pos_weight)
        above_decay = torch.exp(-torch.clamp((scale_scores - tau) / self.upper_temperature, min=0.0))
        far_below_gate = torch.sigmoid((tau - scale_scores - self.lower_band) / self.lower_temperature)
        below_factor = 1.0 - self.far_below_suppression * far_below_gate
        scale = below_factor * above_decay
        if self.hard_positive_boost > 0:
            hard_gate = torch.sigmoid((tau + self.lower_band - scale_scores) / self.lower_temperature)
            scale = scale * (1.0 + self.hard_positive_boost * hard_gate)
        return torch.clamp(scale, min=self.min_pos_weight, max=self.max_pos_weight)

    def _subsample_history(self, prev_targets: torch.Tensor, prev_decoys: torch.Tensor, device: torch.device):
        target_prob = min(1.0, self.max_prev_targets / max(1, len(prev_targets)))
        decoy_prob = min(1.0, self.max_prev_decoys / max(1, len(prev_decoys)))
        target_mask = torch.rand(len(prev_targets), generator=self._cpu_rng) < target_prob
        decoy_mask = torch.rand(len(prev_decoys), generator=self._cpu_rng) < decoy_prob
        sampled_targets = prev_targets[target_mask.to(prev_targets.device)]
        sampled_decoys = prev_decoys[decoy_mask.to(prev_decoys.device)]
        return (sampled_targets, sampled_decoys)

    @staticmethod
    def _cap_normalizer(cap: int) -> float:
        return float(max(1, cap))

    def _similarity_scores(self, query_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor):
        batch_size = query_emb.size(0)
        device = query_emb.device
        sim_pos_all = query_emb @ pos_emb.T
        sim_neg_all = query_emb @ neg_emb.T
        diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        if self.score_mode == 'query_centered':
            row_center = sim_neg_all.mean(dim=1, keepdim=True)
            sim_pos_for_scores = sim_pos_all - row_center
            sim_neg_for_scores = sim_neg_all - row_center
        elif self.score_mode == 'positive_gap':
            row_pos = sim_pos_all.diag().unsqueeze(1)
            sim_pos_for_scores = sim_pos_all - row_pos + sim_pos_all.diag().unsqueeze(1)
            sim_neg_for_scores = sim_neg_all - row_pos
        else:
            sim_pos_for_scores = sim_pos_all
            sim_neg_for_scores = sim_neg_all
        return (sim_pos_for_scores, sim_neg_for_scores, sim_pos_all, sim_neg_all, diag_mask)

    def _row_decoys(self, sim_pos_for_scores: torch.Tensor, sim_neg_for_scores: torch.Tensor, diag_mask: torch.Tensor) -> torch.Tensor:
        batch_size = sim_pos_for_scores.size(0)
        if self.decoy_mode == 'paired':
            return sim_neg_for_scores.diag().view(batch_size, 1)
        if self.decoy_mode == 'explicit_all':
            return sim_neg_for_scores
        other_pos = sim_pos_for_scores.masked_select(~diag_mask).view(batch_size, batch_size - 1)
        return torch.cat([sim_neg_for_scores, other_pos], dim=1)

    def _select_hard_decoys(self, decoys: torch.Tensor) -> torch.Tensor:
        if self.hard_decoy_topk <= 0 or self.hard_decoy_topk >= decoys.size(1):
            return decoys
        return torch.topk(decoys, k=self.hard_decoy_topk, dim=1).values

    def _current_rank_loss(self, target_scores: torch.Tensor, row_decoys: torch.Tensor) -> torch.Tensor:
        if target_scores.numel() == 0 or row_decoys.numel() == 0:
            return torch.sum(target_scores) * 1e-09
        selected_decoys = self._select_hard_decoys(row_decoys)
        if self.prev_scores is not None and self.prev_labels is not None:
            prev_decoys = self.prev_scores.to(target_scores.device)[self.prev_labels.to(target_scores.device) < 0.5]
        else:
            prev_decoys = row_decoys.detach().reshape(-1)
        adaptive_scale = self._compute_adaptive_scale(target_scores, prev_decoys)
        margin = torch.clamp(self.margin, min=0.0)
        with torch.no_grad():
            mean = row_decoys.detach().mean(dim=1)
            std = row_decoys.detach().std(dim=1, unbiased=False).clamp_min(self.eps)
            quantile_boundary = torch.quantile(row_decoys.detach().float(), q=self.fallback_quantile, dim=1)
            adaptive_boundary = 0.5 * quantile_boundary + 0.5 * (mean + self.constraint_value * std)
            band_gate = torch.exp(-torch.abs(selected_decoys.detach() - adaptive_boundary.unsqueeze(1)) / max(self.lower_band, self.eps))
            violation_gate = torch.sigmoid((selected_decoys.detach() - target_scores.unsqueeze(1) + margin) / self.lower_temperature)
            weights = 1.0 + self.boundary_decoy_weight * band_gate * violation_gate
            target_gate = torch.sigmoid((adaptive_boundary + margin - target_scores.detach()) / self.lower_temperature)
            target_weights = self.min_pos_weight + (self.max_pos_weight - self.min_pos_weight) * target_gate
            active_rate = (selected_decoys.detach() - target_scores.detach().unsqueeze(1) + margin.detach() > 0).float().mean()
            boundary_rate = ((band_gate > torch.exp(torch.tensor(-1.0, device=band_gate.device))) & (violation_gate > 0.5)).float().mean()
            self._epoch_active_comparison_sum += float(active_rate.cpu())
            self._epoch_boundary_selection_sum += float(boundary_rate.cpu())
            self._epoch_current_batches += 1
        pos_logits = target_scores.unsqueeze(1)
        decoy_logits = selected_decoys + margin + self.soft_rank_temperature * torch.log(weights.clamp_min(self.eps))
        listwise_loss = -F.log_softmax(torch.cat([pos_logits, decoy_logits], dim=1) / self.soft_rank_temperature, dim=1)[:, 0]
        boundary_gap = F.softplus((adaptive_boundary + margin - target_scores) / self.soft_rank_temperature) * self.soft_rank_temperature
        per_query = listwise_loss + boundary_gap
        return (per_query * adaptive_scale * target_weights).mean()

    def _arl_loss(self, current_scores: torch.Tensor, current_labels: torch.Tensor) -> torch.Tensor:
        if self.prev_scores is None or self.prev_labels is None:
            self.initialize_history(current_labels)
        device = current_scores.device
        prev_scores = self.prev_scores.to(device)
        prev_labels = self.prev_labels.to(device)
        current_is_target = current_labels >= 0.5
        prev_is_target = prev_labels >= 0.5
        if torch.sum(current_is_target) == 0 or torch.sum(~current_is_target) == 0:
            return torch.sum(current_scores) * 1e-09
        current_targets = current_scores[current_is_target]
        current_decoys = current_scores[~current_is_target]
        prev_targets = prev_scores[prev_is_target]
        prev_decoys = prev_scores[~prev_is_target]
        if prev_targets.numel() == 0 or prev_decoys.numel() == 0:
            return torch.sum(current_scores) * 1e-09
        adaptive_scale = self._compute_adaptive_scale(current_targets, prev_decoys)
        sampled_prev_targets, sampled_prev_decoys = self._subsample_history(prev_targets, prev_decoys, device)
        margin = torch.clamp(self.margin, min=0.0)
        loss_components = []
        if current_targets.numel() > 0 and sampled_prev_decoys.numel() > 0:
            diff_matrix = sampled_prev_decoys.view(1, -1) - current_targets.view(-1, 1) + margin
            squared_loss = torch.clamp(diff_matrix, min=0) ** 2
            loss_components.append(torch.sum(squared_loss * adaptive_scale.view(-1, 1)) / self._cap_normalizer(self.max_prev_targets))
        if current_decoys.numel() > 0 and sampled_prev_targets.numel() > 0:
            diff_matrix = current_decoys.view(-1, 1) - sampled_prev_targets.view(1, -1) + margin
            squared_loss = torch.clamp(diff_matrix, min=0) ** 2
            loss_components.append(torch.sum(squared_loss) / self._cap_normalizer(self.max_prev_decoys))
        total_loss = sum(loss_components) if loss_components else torch.tensor(0.0, device=device)
        final_loss = torch.where(torch.isnan(total_loss), torch.zeros_like(total_loss), total_loss)
        if not final_loss.requires_grad:
            final_loss = final_loss + 0.0 * (current_scores.sum() + self.margin)
        return final_loss

    def forward(self, query_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> LossOutput:
        sim_pos_for_scores, sim_neg_for_scores, _, _, diag_mask = self._similarity_scores(query_emb, pos_emb, neg_emb)
        target_scores = sim_pos_for_scores.diag()
        row_decoys = self._row_decoys(sim_pos_for_scores, sim_neg_for_scores, diag_mask)
        decoy_scores = row_decoys.reshape(-1)
        current_scores = torch.cat([target_scores, decoy_scores], dim=0)
        current_labels = torch.cat([torch.ones_like(target_scores), torch.zeros_like(decoy_scores)], dim=0)
        self._epoch_scores.append(current_scores.detach().cpu())
        self._epoch_labels.append(current_labels.detach().cpu())
        history_loss = self._arl_loss(current_scores, current_labels)
        current_loss = self._current_rank_loss(target_scores, row_decoys)
        loss = self.history_pair_weight * history_loss + self.current_pair_weight * current_loss
        with torch.no_grad():
            prev_targets = int(torch.sum(self.prev_labels >= 0.5).item()) if self.prev_labels is not None else 0
            prev_decoys = int(torch.sum(self.prev_labels < 0.5).item()) if self.prev_labels is not None else 0
            stats = {'loss': float(loss.detach().cpu()), 'pos_sim_mean': float(target_scores.detach().mean().cpu()), 'neg_sim_mean': float(decoy_scores.detach().mean().cpu()), 'arl_history_loss': float(history_loss.detach().cpu()), 'arl_current_loss': float(current_loss.detach().cpu()), 'margin': float(torch.clamp(self.margin.detach().cpu(), min=0.0)), 'tau': self.get_cached_tau(), 'history_updates': float(self.history_update_count), 'prev_targets': float(prev_targets), 'prev_decoys': float(prev_decoys), 'decoy_mode': self.decoy_mode, 'score_mode': self.score_mode, 'hard_decoy_topk': float(self.hard_decoy_topk), 'soft_rank_temperature': self.soft_rank_temperature, 'boundary_decoy_weight': self.boundary_decoy_weight, 'max_pos_weight': self.max_pos_weight, 'history_pair_weight': self.history_pair_weight, 'current_pair_weight': self.current_pair_weight}
        return (loss, stats)

def create_loss(name, config):
    if name.lower() not in ('arl', 'boro'):
        raise ValueError('Expected BORO')
    return AdaptiveRankLossIR(**config.get('loss_params', {}).get('arl', {}))

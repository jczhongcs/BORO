import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import joblib
from sklearn.metrics import roc_auc_score
from .loss import AdaptiveRankLoss
input_size = 8
hidden_size = 64
num_classes = 1
num_epochs = 50
batch_size = 100
num_blocks = 9
cardinality = 8
seed = 8256
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
LOSS_METHOD_SOURCES = {'boro': {'display_name': 'BORO', 'source': 'boro.prsm.loss', 'url': '', 'requires': []}}

def set_global_seed(value):
    random.seed(value)
    torch.manual_seed(value)
    np.random.seed(value)
DEFAULT_LOSS_CONFIG = {'max_prev_targets': 5000, 'max_prev_decoys': 5000, 'constraint_value': 0.01, 'lower_band': 0.26, 'min_pos_weight': 0.35, 'upper_temperature': 1.0, 'lower_temperature': 0.1, 'far_below_suppression': 0.1, 'boundary_start_epoch': 3, 'threshold_momentum': 0.5, 'fallback_quantile': 0.99, 'exact_scan_max_items': 200000, 'num_bins': 4096, 'verbose': False, 'history_pair_weight': 1.0, 'current_pair_weight': 0.003, 'current_hard_negatives': 4, 'soft_rank_temperature': 0.6, 'violation_temperature': 0.1, 'boundary_negative_weight': 0.5, 'boundary_gap_weight': 0.15}

class TrainingLossAdapter:
    tau_value = None

    def initialize_history(self, labels):
        return None

    def __call__(self, predictions, labels, model, indices=None):
        raise NotImplementedError

    def end_epoch(self, predictions, labels):
        return {}

    def metadata(self):
        return {}

class BOROLossAdapter(TrainingLossAdapter):

    def __init__(self, loss_config):
        self.loss_config = loss_config
        self.criterion = AdaptiveRankLoss(**loss_config)

    def initialize_history(self, labels):
        self.criterion.initialize_history(labels)

    def __call__(self, predictions, labels, model, indices=None):
        return self.criterion(current_scores=predictions, current_labels=labels.float(), margin=model.margin)

    def end_epoch(self, predictions, labels):
        self.criterion.update_history(predictions.float(), labels.float())
        self.tau_value = self.criterion.get_cached_tau()
        return {'tau': self.tau_value}

class ResidualBlock(nn.Module):

    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual
        return out

class ResNeXtBlock(nn.Module):

    def __init__(self, in_features, out_features, cardinality=32):
        super(ResNeXtBlock, self).__init__()
        self.cardinality = cardinality
        self.group_width = out_features // self.cardinality
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.fc3 = nn.Linear(out_features, out_features)
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(nn.Linear(in_features, out_features))

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out += self.shortcut(residual)
        return out

class ResNeXt(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, num_blocks, cardinality=8):
        super(ResNeXt, self).__init__()
        self.fc_input = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.residual_blocks = nn.ModuleList([ResNeXtBlock(hidden_size, hidden_size, cardinality) for _ in range(num_blocks)])
        self.fc_output = nn.Linear(hidden_size, 1)
        self.margin = nn.Parameter(torch.tensor(0.2), requires_grad=True)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.fc_input(x)
        x = self.relu(x)
        for block in self.residual_blocks:
            x = block(x)
        return torch.sigmoid(self.fc_output(x)).squeeze(-1)

def build_optimizer(model, loss_method, criterion, optimizer_method='adam'):
    return optim.Adam([{'params': [p for p in model.parameters() if p is not model.margin], 'lr': 0.001}, {'params': [model.margin], 'lr': 1e-5}])

def build_training_loss(loss_method, loss_config, data_len=None, pos_len=None):
    if loss_method != 'boro':
        raise ValueError('Expected BORO')
    return BOROLossAdapter(loss_config)

def maybe_update_optimizer(optimizer, loss_method, optimizer_method, epoch):
    return None

def train_model(dataset, model_output_path, loss_config, history_dir, loss_method='boro', optimizer_method='adam'):
    if loss_method != 'boro':
        raise ValueError('Expected BORO')
    data = dataset
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    all_y = []
    for batch in train_loader:
        _, y, _ = batch
        all_y.extend(y.numpy())
    torch.rand(len(all_y)).float().cpu()
    model = ResNeXt(input_size, hidden_size, num_classes, num_blocks, cardinality).cpu()
    loss_config = loss_config or DEFAULT_LOSS_CONFIG.copy()
    loss_source = LOSS_METHOD_SOURCES[loss_method]
    print('loss_method:', loss_method)
    print('loss_source:', loss_source)
    print('optimizer_method:', optimizer_method)
    if loss_method == 'boro':
        print('AdaptiveRankLoss参数:', loss_config)
    criterion = build_training_loss(loss_method, loss_config, data_len=len(data), pos_len=int(np.sum(np.asarray(data.label) == 1)))
    criterion_metadata = criterion.metadata()
    if criterion_metadata:
        print('criterion_metadata:', criterion_metadata)
    criterion.initialize_history(all_y)
    optimizer = build_optimizer(model, loss_method, criterion, optimizer_method)
    print('optimizer:', optimizer.__class__.__name__)
    auc_history = []
    loss_history = []
    margin_history = []
    threshold_history = []
    for epoch in range(num_epochs):
        maybe_update_optimizer(optimizer, loss_method, optimizer_method, epoch)
        model.train()
        epoch_predictions = []
        epoch_true_labels = []
        epoch_total_loss = 0
        for i, batch in enumerate(train_loader):
            batch_inputs, batch_labels, batch_indices = batch
            batch_inputs = batch_inputs.float().cpu()
            batch_labels = batch_labels.float().cpu()
            batch_indices = batch_indices.long().cpu()
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_labels, model, batch_indices)
            loss.backward()
            optimizer.step()
            model.margin.data = torch.clamp(model.margin.data, min=0.0)
            epoch_predictions.extend(predictions.detach().cpu().numpy())
            epoch_true_labels.extend(batch_labels.detach().cpu().numpy())
            epoch_total_loss += loss.item()
        all_preds_tensor = torch.tensor(epoch_predictions).float().cpu()
        all_labels_tensor = torch.tensor(epoch_true_labels).float().cpu()
        epoch_extra = criterion.end_epoch(all_preds_tensor, all_labels_tensor)
        tau_value = epoch_extra.get('tau', float('nan'))
        threshold_history.append(tau_value)
        avg_loss = epoch_total_loss / len(train_loader)
        current_auc = roc_auc_score(all_labels_tensor.numpy(), all_preds_tensor.numpy())
        auc_history.append(current_auc)
        loss_history.append(avg_loss)
        margin_history.append(model.margin.item())
        print(f'Epoch [{epoch + 1}/{num_epochs}] AUC: {current_auc:.4f}, Margin: {model.margin.item():.4f}, Tau: {tau_value:.6f}, Avg Loss: {avg_loss:.4f}')
    try:
        os.makedirs(history_dir, exist_ok=True)
        margin_file_path = os.path.join(history_dir, f'margin_history_seed{seed}.tsv')
        with open(margin_file_path, 'w', encoding='utf-8') as f_out:
            f_out.write('Epoch\tMargin\n')
            for idx, m in enumerate(margin_history, 1):
                f_out.write(f'{idx}\t{m:.6f}\n')
        print(f'已保存margin历史到: {margin_file_path}')
        threshold_file_path = os.path.join(history_dir, f'threshold_history_seed{seed}.tsv')
        with open(threshold_file_path, 'w', encoding='utf-8') as f_out:
            f_out.write('Epoch\tOperatingThresholdTau\n')
            for idx, tau in enumerate(threshold_history, 1):
                f_out.write(f'{idx}\t{tau:.6f}\n')
        print(f'已保存operating threshold历史到: {threshold_file_path}')
    except Exception as e:
        print(f'保存margin历史失败: {e}')
    os.makedirs(os.path.dirname(model_output_path) or '.', exist_ok=True)
    joblib.dump(model, model_output_path)
    history_payload = {'seed': seed, 'loss_method': loss_method, 'optimizer_method': optimizer_method, 'loss_source': loss_source, 'loss_config': loss_config, 'criterion_metadata': criterion_metadata, 'auc_history': auc_history, 'loss_history': loss_history, 'margin_history': margin_history, 'threshold_history': threshold_history}
    joblib.dump(history_payload, os.path.join(os.path.dirname(model_output_path) or '.', 'training_history.pkl'))
    print(f'保存DP模型: {model_output_path}')

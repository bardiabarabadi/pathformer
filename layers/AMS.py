import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from layers.Layer import Transformer_Layer
from utils.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP


class AMS(nn.Module):
    def __init__(self, input_size, output_size, num_experts, device, num_nodes=1, d_model=32, d_ff=64, dynamic=False,
                 patch_size=[8, 6, 4, 2], noisy_gating=True, k=4, layer_number=1, residual_connection=1, batch_norm=False):
        super(AMS, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k

        self.start_linear = nn.Linear(in_features=num_nodes, out_features=1)
        self.seasonality_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        self.experts = nn.ModuleList()
        self.MLPs = nn.ModuleList()
        for patch in patch_size:
            patch_nums = int(input_size / patch)
            self.experts.append(Transformer_Layer(device=device, d_model=d_model, d_ff=d_ff,
                                      dynamic=dynamic, num_nodes=num_nodes, patch_nums=patch_nums,
                                      patch_size=patch, factorized=True, layer_number=layer_number, batch_norm=batch_norm))

        # self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Linear(input_size, num_experts)
        self.w_gate = nn.Linear(input_size, num_experts)

        self.residual_connection = residual_connection
        self.end_MLP = MLP(input_size=input_size, output_size=output_size)

        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_positions_if_in = torch.clamp(threshold_positions_if_in, max=top_values_flat.numel() - 1)
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = torch.clamp(threshold_positions_if_in - 1, min=0)
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)

        safe_noise_stddev = torch.clamp(noise_stddev, min=1e-6)
        safe_noise_stddev = torch.nan_to_num(safe_noise_stddev, nan=1e-6, posinf=1e6, neginf=1e-6)

        normalized_if_in = (clean_values - threshold_if_in) / safe_noise_stddev
        normalized_if_out = (clean_values - threshold_if_out) / safe_noise_stddev

        normalized_if_in = torch.nan_to_num(normalized_if_in, nan=0.0, posinf=10.0, neginf=-10.0)
        normalized_if_out = torch.nan_to_num(normalized_if_out, nan=0.0, posinf=10.0, neginf=-10.0)

        normalized_if_in = torch.clamp(normalized_if_in, min=-10.0, max=10.0)
        normalized_if_out = torch.clamp(normalized_if_out, min=-10.0, max=10.0)

        prob_if_in = normal.cdf(normalized_if_in)
        prob_if_out = normal.cdf(normalized_if_out)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def seasonality_and_trend_decompose(self, x):
        x = x[:, :, :, 0]
        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        decomposed = x + seasonality + trend
        return torch.nan_to_num(decomposed, nan=0.0, posinf=0.0, neginf=0.0)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        x = self.start_linear(x).squeeze(-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # clean_logits = x @ self.w_gate
        clean_logits = self.w_gate(x)
        clean_logits = torch.nan_to_num(clean_logits, nan=0.0, posinf=0.0, neginf=0.0)
        if self.noisy_gating and train:
            # raw_noise_stddev = x @ self.w_noise
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noise_stddev = torch.nan_to_num(noise_stddev, nan=noise_epsilon, posinf=1e6, neginf=noise_epsilon)
            noise_stddev = torch.clamp(noise_stddev, min=noise_epsilon)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            noisy_logits = torch.nan_to_num(noisy_logits, nan=0.0, posinf=0.0, neginf=0.0)
            logits = noisy_logits
        else:
            logits = clean_logits
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        top_k_gates = torch.nan_to_num(top_k_gates, nan=0.0, posinf=0.0, neginf=0.0)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        gates = torch.nan_to_num(gates, nan=0.0, posinf=0.0, neginf=0.0)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        new_x = self.seasonality_and_trend_decompose(x)

        #multi-scale router
        gates, load = self.noisy_top_k_gating(new_x, self.training)
        # calculate balance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[i](expert_inputs[i])[0] for i in range(self.num_experts)]
        output = dispatcher.combine(expert_outputs)
        if self.residual_connection:
            output = output + x
        return output, balance_loss






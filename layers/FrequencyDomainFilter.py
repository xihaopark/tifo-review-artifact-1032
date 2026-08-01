import torch
import torch.nn.functional as F

class FrequencyDomainFilter:
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def _compute_global_statistics(self, train_loader):
        num_channels = self.args.enc_in
        freq_length = self.args.seq_len // 2 + 1

        mean_xf_real = torch.zeros(freq_length, num_channels, dtype=torch.float32).to(self.device)
        var_xf_real = torch.zeros(freq_length, num_channels, dtype=torch.float32).to(self.device)
        mean_xf_imag = torch.zeros(freq_length, num_channels, dtype=torch.float32).to(self.device)
        var_xf_imag = torch.zeros(freq_length, num_channels, dtype=torch.float32).to(self.device)
        count = 0

        with torch.no_grad():
            for data in train_loader:
                lookback_window = data[0].float().to(self.device)
                for ch in range(num_channels):
                    xf = torch.fft.rfft(lookback_window[:, :, ch], dim=1)

                    mean_xf_real[:, ch] += xf.real.mean(dim=0)
                    var_xf_real[:, ch] += xf.real.var(dim=0, unbiased=False)
                    mean_xf_imag[:, ch] += xf.imag.mean(dim=0)
                    var_xf_imag[:, ch] += xf.imag.var(dim=0, unbiased=False)
                count += xf.size(0)

        mean_xf_real /= count
        var_xf_real /= count
        mean_xf_imag /= count
        var_xf_imag /= count

        return mean_xf_real, torch.sqrt(var_xf_real + 1e-5), mean_xf_imag, torch.sqrt(var_xf_imag + 1e-5)

    def precompute_local_stability(self, train_loader, sample_rate=0.1):
        num_channels = self.args.enc_in
        freq_length = self.args.seq_len // 2 + 1

        local_stability_real = torch.zeros(freq_length, num_channels, dtype=torch.float32).to(self.device)
        local_stability_imag = torch.zeros(freq_length, num_channels, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            sample_indices = torch.randperm(len(train_loader))[:int(sample_rate * len(train_loader))]
            for idx in sample_indices:
                data = train_loader[idx]
                lookback_window, target = data[0].float().to(self.device), data[1].float().to(self.device)
                for ch in range(num_channels):
                    xf = torch.fft.rfft(lookback_window[:, :, ch], dim=1)
                    yf = torch.fft.rfft(target[:, :, ch], dim=1)

                    min_length = min(xf.size(1), yf.size(1))
                    xf_truncated = xf[:, :min_length]
                    yf_truncated = yf[:, :min_length]

                    cov_real = torch.mean((xf_truncated.real - xf_truncated.real.mean(dim=0, keepdim=True)) * (yf_truncated.real - yf_truncated.real.mean(dim=0, keepdim=True)), dim=0)
                    cov_imag = torch.mean((xf_truncated.imag - xf_truncated.imag.mean(dim=0, keepdim=True)) * (yf_truncated.imag - yf_truncated.imag.mean(dim=0, keepdim=True)), dim=0)

                    local_stability_real[:, ch] += cov_real
                    local_stability_imag[:, ch] += cov_imag

        local_stability_real /= len(sample_indices)
        local_stability_imag /= len(sample_indices)

        return local_stability_real, local_stability_imag

    def combine_stabilities(self, global_mask_real, global_mask_imag, local_stability_real, local_stability_imag, alpha=0.5):
        combined_mask_real = alpha * global_mask_real + (1 - alpha) * local_stability_real
        combined_mask_imag = alpha * global_mask_imag + (1 - alpha) * local_stability_imag
        return combined_mask_real, combined_mask_imag

    def dynamic_alpha(self, volatility, local_stability):
        return 1 / (1 + volatility / (local_stability + 1e-5))

    def _apply_mask(self, batch_x, global_mask_real, global_mask_imag, local_stability_real, local_stability_imag):
        means = batch_x.mean(1, keepdim=True).detach()
        batch_x = batch_x - means
        stdev = torch.sqrt(torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        batch_x /= stdev

        batch_x_fft = torch.fft.rfft(batch_x, dim=1)
        global_mask_real_expanded = global_mask_real.unsqueeze(0).unsqueeze(2).expand_as(batch_x_fft.real)
        global_mask_imag_expanded = global_mask_imag.unsqueeze(0).unsqueeze(2).expand_as(batch_x_fft.imag)

        masked_fft_global_real = batch_x_fft.real * global_mask_real_expanded
        masked_fft_global_imag = batch_x_fft.imag * global_mask_imag_expanded

        combined_mask_real, combined_mask_imag = self.combine_stabilities(global_mask_real, global_mask_imag, local_stability_real, local_stability_imag)

        masked_fft_final_real = masked_fft_global_real * combined_mask_real
        masked_fft_final_imag = masked_fft_global_imag * combined_mask_imag

        masked_fft_final = torch.complex(masked_fft_final_real, masked_fft_final_imag)

        masked_input_final = torch.fft.irfft(masked_fft_final, dim=1, n=batch_x.size(1))

        masked_input_final = masked_input_final * stdev[:, 0, :].unsqueeze(1).repeat(1, self.args.seq_len, 1)
        masked_input_final = masked_input_final + means[:, 0, :].unsqueeze(1).repeat(1, self.args.seq_len, 1)

        return masked_input_final

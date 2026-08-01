import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
class GlobalMaskCalculator:
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def compute_global_statistics(self, train_loader):
        num_channels = self.args.enc_in
        freq_length = self.args.seq_len #// 2 + 1

        amplitude_sum = torch.zeros(freq_length, num_channels, dtype=torch.float32).to(self.device)
        amplitude_squared_sum = torch.zeros(freq_length, num_channels, dtype=torch.float32).to(self.device)
        count = 0

        with torch.no_grad():
            for data in train_loader:
                lookback_window = data[0].float().to(self.device)
                xf = torch.fft.fft(lookback_window, dim=1)
                amplitude = torch.abs(xf)

                amplitude_sum += amplitude.sum(dim=0)
                amplitude_squared_sum += (amplitude ** 2).sum(dim=0)
                count += lookback_window.size(0)

        mean_amplitude = amplitude_sum / count
        mean_amplitude_squared = amplitude_squared_sum / count
        variance_amplitude = mean_amplitude_squared - (mean_amplitude ** 2)
        std_amplitude = torch.sqrt(variance_amplitude + 1e-5)

        global_mask_amp = mean_amplitude / (std_amplitude + 1e-5)

        return global_mask_amp

def run_filter(args, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculator = GlobalMaskCalculator(args, device)
    global_mask_amp = calculator.compute_global_statistics(train_loader)
    return global_mask_amp


####
####
####

import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_difference(data, n=1):
    """
    Apply differencing to the data.
    :param data: Input data [batch, length, channel]
    :param n: Order of differencing
    :return: Differenced data and the last original data point for each series
    """
    for i in range(n):
        data[:, 1:, :] = data[:, 1:, :] - data[:, :-1, :]
    return data

class FrequencyDomainFilter(nn.Module):
    def __init__(self, args, global_mask_amp):
        super(FrequencyDomainFilter, self).__init__()
        self.args = args
        num_channels = args.enc_in
        self.output_length = args.pred_len
        self.mask = global_mask_amp

        # Global filter linear layers with L2 regularization and Dropout
        self.linear_r = nn.Sequential(
            nn.Linear(global_mask_amp.size(0), 512),  # bottleneck structure
            nn.RReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, global_mask_amp.size(0)),
        )

        self.linear_i = nn.Sequential(
            nn.Linear(global_mask_amp.size(0), 512),  # bottleneck structure
            nn.RReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, global_mask_amp.size(0)),
        )

        input  = self.args.seq_len * num_channels
        hidden = 32  # Increased hidden size for better nonlinearity

        # MLP with bottleneck and regularization for trend prediction
        self.trend_predictor = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, hidden // 2),  # Bottleneck layer
            nn.ReLU(),
            nn.Linear(hidden // 2, self.output_length * num_channels)
        )

    def apply_global_filter(self, data):
        data_fft = torch.fft.fft(data, dim=1)
        mask_r = self.linear_r(self.mask.permute(1, 0)).permute(1, 0)
        mask_i = self.linear_i(self.mask.permute(1, 0)).permute(1, 0)
        masked_fft_real = data_fft.real * mask_r
        masked_fft_imag = data_fft.imag * mask_i
        masked_fft = torch.complex(masked_fft_real, masked_fft_imag)
        filtered_data = torch.fft.ifft(masked_fft, dim=1, n=data.size(1)).real
        return filtered_data

    def inverse_filter(self, data, eps=1e-5, reg=1e-3):
        F = torch.fft.fft(data, dim=1)
        with torch.no_grad():
            m_r = self.linear_r(self.mask.T).T
            m_i = self.linear_i(self.mask.T).T
        inv_r = m_r.conj() / (m_r.pow(2) + reg + eps)
        inv_i = m_i.conj() / (m_i.pow(2) + reg + eps)
        F_inv = torch.complex(F.real * inv_r, F.imag * inv_i)
        return torch.fft.ifft(F_inv, dim=1, n=data.size(1)).real


    def forward(self, batch_x):
        #differenced_data = apply_difference(batch_x, n=1)
        differenced_data = batch_x #ablation study
        filtered_data = self.apply_global_filter(differenced_data)

        return filtered_data

import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout, use_group_norm: bool):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(1, out_ch) if use_group_norm else nn.Identity(),
            nn.ReLU(),
            Chomp1d(pad),
            nn.Dropout(dropout),

            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(1, out_ch) if use_group_norm else nn.Identity(),
            nn.ReLU(),
            Chomp1d(pad),
            nn.Dropout(dropout),
        ]
        self.net = nn.Sequential(*layers)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, layers: int, kernel_size: int, dropout: float, use_group_norm: bool):
        super().__init__()
        blocks = []
        ch_in = in_channels
        for i in range(layers):
            dilation = 2 ** i
            blocks.append(TemporalBlock(ch_in, hidden, kernel_size, dilation, dropout, use_group_norm))
            ch_in = hidden
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        z = self.tcn(x)
        return self.head(z)

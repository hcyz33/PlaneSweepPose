import torch.nn as nn
import torch

class PoseCNN(nn.Module):
    def __init__(self, num_joints, num_bones, hidden_size, output_size):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv1d(num_joints, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.input_layer_2 = nn.Sequential(
            nn.Conv1d(num_bones, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
            ) for _ in range(2)
        ])
        self.output_layer = nn.Conv1d(hidden_size, output_size, kernel_size=1, bias=False)

    def forward(self, x, x2):
        """
        Args
            x: [B, Nj, Nd]
        Returns
            [B, Cout, Nd]
        """
        x = self.input_layer(x)
        # x = x + self.input_layer_2(x2)
        for l in range(len(self.res_layers)):
            y = self.res_layers[l](x)
            x = x + y
        x = self.output_layer(x)

        return x


class JointCNN(nn.Module):
    def __init__(self, num_joints, num_bones, hidden_size, output_size):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv1d(num_joints, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.input_layer_2 = nn.Sequential(
            nn.Conv1d(num_bones, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.channel_att = ChannelAtt(hidden_size, hidden_size//2)
        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, dilation=1, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=8, dilation=8, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=16, dilation=16, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=32, dilation=32, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, dilation=1, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                ChannelAtt(hidden_size, hidden_size//2)
            ) for _ in range(2)
        ])
        self.output_layer = nn.Conv1d(hidden_size, output_size, kernel_size=1, bias=False)

    def forward(self, x, x2):
        """
        Args
            x: [B, Nj, Nd]
        Returns
            [B, Cout, Nd]
        """
        x = self.input_layer(x)
        # x = x + self.input_layer_2(x2)
        x = self.channel_att(x)
        for l in range(len(self.res_layers)):
            y = self.res_layers[l](x)
            x = x + y
        x = self.output_layer(x)

        return x

class ChannelAtt(nn.Module):
    def __init__(self, channel_size, hidden_size):
        super().__init__()

        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.MLP =  nn.Sequential(
                nn.Conv1d(channel_size, hidden_size, kernel_size=1, bias=False),
                nn.Conv1d(hidden_size, channel_size, kernel_size=1, bias=False)
            )
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.Conv1d(2, 1, 1, padding=0, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的

    def forward(self, x):
        """
        Args
            x: [B, Nj, Nd]
        Returns
            [B, Cout, Nd]
        """
        x_average = self.average_pool(x)
        x_maxpool = self.max_pool(x)
        # x = x + self.input_layer_2(x2)
        x_average = self.MLP(x_average)
        x_maxpool = self.MLP(x_maxpool)
        att = x_average + x_maxpool
        att = self.sigmoid(att)
        out = x * att

        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.conv1(att)  # 对池化完的数据cat 然后进行卷积
        att = self.sigmoid(att)
        out = x * att
        return out


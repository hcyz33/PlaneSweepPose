import torch.nn as nn


class PoseRegressNet(nn.Module):
    def __init__(self, num_joints, hidden_size, output_size):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv1d(num_joints, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
            ) for _ in range(2)
        ])
        self.output_layer = nn.Conv1d(hidden_size, output_size, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Args
            x: [B, Nj*2, 1]
        Returns
            [B, NJ, 1] rougth depth of joints reletive to root joint
        """
        x = self.input_layer(x)
        for l in range(len(self.res_layers)):
            y = self.res_layers[l](x)
            x = x + y
        x = self.output_layer(x)

        return x



import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

ACT = nn.GELU
# ACT = nn.SiLU

class PosEmbed2D(nn.Module):
    def __init__(self,
                 grid_size=(23, 23),
                 mode="grid",
                 fourier_dim=2):
        super().__init__()
        H, W = grid_size
        y, x = torch.linspace(0, 1, H), torch.linspace(0, 1, W)
        yy, xx = torch.meshgrid(y, x, indexing="ij")         # [H,W]

        pe = torch.stack((xx, yy), 0)                    # [2,H,W]

        self.register_buffer("pe", pe.unsqueeze(0))          # [1,C,H,W]

    def forward(self, x):                                     # x: [B,C,H,W]
        return torch.cat((x, self.pe.expand_as(x[:, :self.pe.size(1)])), dim=1)


class LearnablePosEmbed2D(nn.Module):
    def __init__(self, grid_size=(23, 23)):
        super().__init__()
        H, W = grid_size
        self.pe = nn.Parameter(torch.randn(1, 2, H, W))

    def forward(self, x):                                     # x: [B,C,H,W]
        return torch.cat((x, self.pe.expand(x.size(0), -1, -1, -1)), dim=1)


class DepthSepBlock(nn.Module):
    def __init__(self, ch, ch_out, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation,
                                groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            ACT(),
            nn.Conv2d(ch, ch_out, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            ACT(),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, ch, ch_out, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch_out, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(ch_out),
            ACT()
        )

    def forward(self, x):
        return self.block(x)


class FaceSolverTiny(nn.Module): # 2.42e-8
    def __init__(self, in_channels=23, pe_mode="grid", depthwise=True):
        super().__init__()
        self.pe_mode = pe_mode
        if pe_mode == 'learnable':
            self.pe = LearnablePosEmbed2D()
        else:
            self.pe = PosEmbed2D(mode=pe_mode)
        if pe_mode != 'none':
            in_channels = in_channels + self.pe.pe.shape[1]

        self.z_weight = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            ACT(),
        )
        if depthwise:
            self.blocks = nn.Sequential(
                DepthSepBlock(16, 16),
                DepthSepBlock(16, 8),
                DepthSepBlock(8, 4, dilation=2),
                DepthSepBlock(4, 2, dilation=3)
            )
        else:
            self.blocks = nn.Sequential(
                ConvBlock(16, 16),
                ConvBlock(16, 8),
                ConvBlock(8, 4, dilation=2),
                ConvBlock(4, 2, dilation=3)
            )

        self.head = nn.Conv2d(2, 1, 1, bias=True)

    def forward(self, x):                # x: [B, 23, 23, 23]
        if self.pe_mode != 'none':
            x = self.pe(x)
        x = self.z_weight(x)             # [B, 8, 23, 23]
        x = self.blocks(x)
        return self.head(x)              # [B, 1, 23, 23]
    

class FaceSolver3D(nn.Module): # 2.42e-8
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Reduce Z: 23 -> 12
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            ACT(),

            # Reduce Z: 12 -> 6
            nn.Conv3d(8, 8, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            ACT(),

            # Reduce Z: 6 -> 3
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            ACT(),

            # Reduce Z: 3 -> 1
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            ACT(),
            nn.Conv3d(32, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
        )


    def forward(self, x):                # x: [B, 23, 23, 23]
        x = x.unsqueeze(1)               # [B, 1, 23, 23, 23]
        x = self.net(x)                  # [B, 1, 1, 23, 23]
        x = x.squeeze(1)                 # [B, 1, 23, 23]
        return x
    

class FaceSolverMLP(nn.Module):
    def __init__(self, in_channels=23):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Linear(23**3, 2048),
            nn.BatchNorm1d(2048),
            ACT(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            ACT(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            ACT(),
            nn.Linear(2048, 23**2),
        )

    def forward(self, x):                # x: [B, 23, 23, 23]
        x = x.view(x.size(0), -1)         # Flatten to [B, 23**3]
        x = self.blocks(x)
        x = x.view(x.size(0), 1, 23, 23)   # Reshape to [B, 1, 23, 23]
        return x

class FacePredictor(nn.Module):
    def __init__(self, depth=23, height=23, width=23, in_channels=1, name="BeefedFacePredictor", head_mode="relu", pe_mode="grid", solver="depthwise"):
        super().__init__()
        self.name = name
        if solver == "mlp":
            self.solver = FaceSolverMLP()
        elif solver == "depthwise":
            self.solver = FaceSolverTiny(pe_mode=pe_mode, depthwise=True)
        elif solver == "3d":
            self.solver = FaceSolver3D()
        else:
            self.solver = FaceSolverTiny(pe_mode=pe_mode, depthwise=False)
        self.head_mode = head_mode

    def forward(self, x):
        """
        x: (B, 1, 1, D, H, W)
        """
        x = x.squeeze(1).squeeze(1)     # (B, D=23, 23, 23)
        x = self.solver(x)              # (B, 1, 23, 23)
        x = x.squeeze(1)                # (B, 23, 23)
        if self.head_mode == 'relu':
            x = F.relu(x) + 1e-10
            x = x / (x.sum(dim=[1,2], keepdim=True))
        elif self.head_mode == 'abs':
            x = torch.abs(x) + 1e-10
            x = x / (x.sum(dim=[1,2], keepdim=True))
        elif self.head_mode == 'softmax':
            x = F.softmax(x.view(x.size(0), -1), dim=1).view(x.size(0), x.size(1), x.size(2))
        else:
            x = x / (x.abs().sum(dim=[1,2], keepdim=True))
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

ACT = nn.GELU
# ACT = nn.SiLU

class PosEmbed2D(nn.Module):
    def __init__(self, grid_size=(23, 23)):
        super().__init__()
        H, W = grid_size
        y, x = torch.linspace(0, 1, H), torch.linspace(0, 1, W)
        yy, xx = torch.meshgrid(y, x, indexing="ij")         # [H,W]

        pe = torch.stack((xx, yy), 0)                    # [2,H,W]

        self.register_buffer("pe", pe.unsqueeze(0))          # [1,C,H,W]

    def forward(self, x):                                     # x: [B,C,H,W]
        return torch.cat((x, self.pe.expand_as(x[:, :self.pe.size(1)])), dim=1)


class DepthSepBlock(nn.Module):
    def __init__(self, ch, ch_out, dilation=1):
        super().__init__()
        self.skip = ch == ch_out
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


class FaceSolver(nn.Module):
    def __init__(self, in_channels=23, pe_mode="grid"):
        super().__init__()
        self.pe_mode = pe_mode
        self.pe = PosEmbed2D()
        if pe_mode == 'grid':
            in_channels = in_channels + self.pe.pe.shape[1]

        self.z_weight = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            ACT(),
        )

        self.blocks = nn.Sequential(
            DepthSepBlock(16, 16),
            DepthSepBlock(16, 8),
            DepthSepBlock(8, 4, dilation=2),
            DepthSepBlock(4, 2, dilation=3)
        )

        self.head = nn.Conv2d(2, 1, 1, bias=True)

    def forward(self, x):                # x: [B, 23, 23, 23]
        if self.pe_mode == 'grid':
            x = self.pe(x)
        x = self.z_weight(x)             # [B, 8, 23, 23]
        x = self.blocks(x)
        return self.head(x)              # [B, 1, 23, 23]


class FaceSolverPlus(nn.Module):
    def __init__(self, in_channels=23, pe_mode="grid"):
        super().__init__()
        self.pe_mode = pe_mode
        self.pe = PosEmbed2D()
        if pe_mode == 'grid':
            in_channels = in_channels + self.pe.pe.shape[1]

        self.z_weight = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            ACT(),
        )
        self.blocks = nn.Sequential(
            DepthSepBlock(32, 32),
            DepthSepBlock(32, 24, dilation=1),
            DepthSepBlock(24, 24, dilation=1),
            DepthSepBlock(24, 16, dilation=1),
            DepthSepBlock(16, 16, dilation=2),
            DepthSepBlock(16, 8, dilation=2),
            DepthSepBlock(8, 4, dilation=3),
        )

        self.head = nn.Conv2d(4, 1, 1, bias=True)

    def forward(self, x):                # x: [B, 23, 23, 23]
        if self.pe_mode == 'grid':
            x = self.pe(x)
        x = self.z_weight(x)
        x = self.blocks(x)
        return self.head(x)


class FacePredictor(nn.Module):
    def __init__(self, name="BeefedFacePredictor", head_mode="relu", pe_mode="grid"):
        super().__init__()
        self.name = name
        self.solver = FaceSolver(pe_mode=pe_mode)
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
    

class SignedFacePredictor(nn.Module):
    def __init__(self, channel_dim=1, 
                 name="BeefedFacePredictor", head_mode="norm", pe_mode="grid"):
        super().__init__()
        self.name = name
        self.solver = FaceSolverPlus(pe_mode=pe_mode)
        self.head_mode = head_mode
        self.channel_dim = channel_dim

    def _prepare_input(self, x):
        if self.channel_dim == 0:
            return x
        elif self.channel_dim == 1:
            return x.permute(0, 2, 1, 3)
        elif self.channel_dim == 2:
            return x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"channel_dim must be 0,1,2 got {self.channel_dim}")

    def forward(self, x):
        """
        x: (B, 1, 1, D, H, W)
        """
        x = x.squeeze(1).squeeze(1)     # (B, D=23, 23, 23)
        x = self._prepare_input(x)
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
        else: # 'norm'
            # x = F.tanh(x)
            x = x / (x.abs().sum(dim=[1,2], keepdim=True))
        return x

    
class FaceSelector(nn.Module):
    def __init__(self, in_channels=1, name="FaceSelectorUnified_v1"):
        super().__init__()
        self.name = name
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, 4, kernel_size=3, stride=2),
            nn.BatchNorm3d(4),
            nn.GELU(),
            nn.Conv3d(4, 8, kernel_size=3, stride=2),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2),
            nn.BatchNorm3d(16),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        # The input is expected to be (B, 1, 1, 23, 23, 23)
        x = x.squeeze(1)
        features = self.backbone(x)
        logits = self.head(features)
        logits = F.softmax(logits, dim=1)
        return logits
      

class FaceSelectorWeight(nn.Module):
    def __init__(self, in_channels=1, name="FaceSelectorUnified_v1"):
        super().__init__()
        self.name = name
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, stride=2),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 64, kernel_size=3, stride=2),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )

        # The head processes the final feature map to produce logits for each face.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        # The input is expected to be (B, 1, 1, 23, 23, 23)
        x = x.squeeze(1)
        features = self.backbone(x)
        logits = self.head(features)
        face_logits = F.softmax(logits[:, :6], dim=1)
        logits = torch.cat([face_logits, logits[:, 6:]], dim=1)
        return logits
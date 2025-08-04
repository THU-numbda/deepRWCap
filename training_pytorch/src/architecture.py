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


class FaceSolverTiny(nn.Module): # 2.42e-8
    def __init__(self, in_channels=23, pe_mode="grid"):
        super().__init__()
        self.pe_mode = pe_mode
        self.pe = PosEmbed2D(mode=pe_mode)
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


class FaceSolverTiny2Head(nn.Module): # 2.42e-8
    def __init__(self, in_channels=23, pe_mode="grid"):
        super().__init__()
        self.pe_mode = pe_mode
        self.pe = PosEmbed2D(mode=pe_mode)
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

        self.head = nn.Conv2d(2, 2, 1, bias=True)

    def forward(self, x):                # x: [B, 23, 23, 23]
        if self.pe_mode == 'grid':
            x = self.pe(x)
        x = self.z_weight(x)
        x = self.blocks(x)
        return self.head(x)              # [B, 2, 23, 23]


class FaceSolverMedium(nn.Module):
    def __init__(self, in_channels=23, pe_mode="grid", size="medium1"):
        super().__init__()
        self.pe_mode = pe_mode
        self.pe = PosEmbed2D(mode=pe_mode)
        if pe_mode == 'grid':
            in_channels = in_channels + self.pe.pe.shape[1]

        if size == "medium1":
            self.z_weight = nn.Sequential(
                nn.Conv2d(in_channels, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                ACT(),
            )
            self.blocks = nn.Sequential(
                DepthSepBlock(64, 64),
                DepthSepBlock(64, 32, dilation=1),
                DepthSepBlock(32, 32, dilation=1),
                DepthSepBlock(32, 16, dilation=1),
                DepthSepBlock(16, 16, dilation=2),
                DepthSepBlock(16, 8, dilation=2),
                DepthSepBlock(8, 4, dilation=3),
            )
        elif size == "medium2":
            self.z_weight = nn.Sequential(
                nn.Conv2d(in_channels, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                ACT(),
            )
            self.blocks = nn.Sequential(
                DepthSepBlock(64, 64),
                DepthSepBlock(64, 32, dilation=1),
                DepthSepBlock(32, 32, dilation=1),
                DepthSepBlock(32, 16, dilation=1),
                DepthSepBlock(16, 8, dilation=2),
                DepthSepBlock(8, 4, dilation=3),
            )
        else: # size == "medium3":
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


class FaceSolverMedium2Head(nn.Module):
    def __init__(self, in_channels=23, pe_mode="grid"):
        super().__init__()
        self.pe_mode = pe_mode
        self.pe = PosEmbed2D(mode=pe_mode)
        if pe_mode == 'grid':
            in_channels = in_channels + self.pe.pe.shape[1]

        self.z_weight = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            ACT(),
        )

        self.blocks = nn.Sequential(
            DepthSepBlock(64, 64),
            DepthSepBlock(64, 32, dilation=1),
            DepthSepBlock(32, 32, dilation=1),
            DepthSepBlock(32, 16, dilation=1),
            DepthSepBlock(16, 16, dilation=2),
            DepthSepBlock(16, 8, dilation=2),
            DepthSepBlock(8, 4, dilation=3),
        )

        self.head = nn.Conv2d(4, 2, 1, bias=True)

    def forward(self, x):                # x: [B, 23, 23, 23]
        if self.pe_mode == 'grid':
            x = self.pe(x)
        x = self.z_weight(x)
        x = self.blocks(x)
        return self.head(x)

class FacePredictor(nn.Module):
    def __init__(self, depth=23, height=23, width=23, in_channels=1, name="BeefedFacePredictor", head_mode="relu", pe_mode="grid"):
        super().__init__()
        self.name = name
        self.solver = FaceSolverTiny(pe_mode=pe_mode)
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
    def __init__(self, depth=23, height=23, width=23, in_channels=1, channel_dim=1, 
                 name="BeefedFacePredictor", head_mode="relu", pe_mode="grid", size="medium1"):
        super().__init__()
        self.name = name
        self.solver = FaceSolverMedium(pe_mode=pe_mode, size=size)
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


class TwoHeadSignedFacePredictor(nn.Module):
    def __init__(self, depth=23, height=23, width=23, in_channels=1, channel_dim=1, name="BeefedFacePredictor", head_mode="relu", pe_mode="grid"):
        super().__init__()
        self.name = name
        self.solver = FaceSolverMedium2Head(pe_mode=pe_mode)
        self.head_mode = head_mode
        self.channel_dim = channel_dim
        self.sign_tensor = torch.tensor(0.0, dtype=torch.float32)

    def _prepare_input(self, x):
        if self.channel_dim == 0:
            return x
        elif self.channel_dim == 1:
            return x.permute(0, 2, 1, 3)
        elif self.channel_dim == 2:
            return x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"channel_dim must be 0,1,2 got {self.channel_dim}")

    def forward(self, x, inference: bool = True):
        """
        x: (B, 1, 1, D, H, W)
        """
        x = x.squeeze(1).squeeze(1)     # (B, D=23, 23, 23)
        x = self._prepare_input(x)
        x = self.solver(x)              # (B, 2, 23, 23)
        mag = x[:, 0, :, :]              # (B, 23, 23)
        self.sign_tensor = x[:, 1, :, :]            # (B, 23, 23)
        # x = x.squeeze(1)                # (B, 23, 23)
        if self.head_mode == 'relu':
            mag = F.relu(mag) + 1e-10
            mag = mag / (mag.sum(dim=[1,2], keepdim=True))            
        elif self.head_mode == 'abs':
            mag = torch.abs(mag) + 1e-10
            mag = mag / (mag.sum(dim=[1,2], keepdim=True))
        elif self.head_mode == 'softmax':
            mag = F.softmax(mag.view(mag.size(0), -1), dim=1).view(mag.size(0), mag.size(1), mag.size(2))
        else: # 'norm'
            # mag = F.tanh(mag)
            mag = mag / (mag.abs().sum(dim=[1,2], keepdim=True))
        
        if inference:
            mag = mag * torch.sign(self.sign_tensor)
        return mag
    

class PosEmbed3D(nn.Module):
    def __init__(self, grid_size=(23, 23, 23)):
        super().__init__()
        D, H, W = grid_size
        z, y, x = (torch.linspace(0, 1, n) for n in (D, H, W))
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        pe = torch.stack((xx, yy, zz), 0).unsqueeze(0)   # (1, 3, D, H, W)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, C, D, H, W)
        # Broadcast only on the batch dimension
        pos = self.pe.expand(x.size(0), -1, -1, -1, -1)   # (B, 3, D, H, W)
        return torch.cat((x, pos), dim=1)                 # (B, C+3, D, H, W)

class DepthSepBlock3D(nn.Module):
    """
    A 3D depthwise separable convolution block.
    Efficiently processes 3D data by separating spatial and channel-wise convolutions.
    """
    def __init__(self, ch_in, ch_out, stride=1):
        super().__init__()
        # Use a strided convolution in the depthwise layer for downsampling
        self.block = nn.Sequential(
            # Depthwise Convolution (Spatial Filtering)
            nn.Conv3d(ch_in, ch_in, kernel_size=3, stride=stride, padding=0, groups=ch_in, bias=False),
            nn.BatchNorm3d(ch_in),
            ACT(),
            # Pointwise Convolution (Channel Mixing)
            nn.Conv3d(ch_in, ch_out, kernel_size=1, bias=False),
            nn.BatchNorm3d(ch_out),
            ACT(),
        )

    def forward(self, x):
        return self.block(x)
    
class FaceSelectorUnified(nn.Module):
    """
    An efficient 3D face selector using a unified, depthwise-separable architecture.
    """
    def __init__(self, in_channels=1, num_faces=6, name="FaceSelectorUnified_v1", head_mode="relu", pe_mode="grid"):
        super().__init__()
        self.name = name
        self.pe_mode = pe_mode
        self.pe   = PosEmbed3D()
        self.head_mode = head_mode
        if pe_mode == "grid":
            in_channels = in_channels + self.pe.pe.shape[1]
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
            nn.Flatten(),                    # -> (B, 64)
            nn.Linear(128, 64),         # -> (B, 6)
            nn.GELU(),
            nn.Linear(64, num_faces)         # -> (B, 6)
        )

    def forward(self, x):
        # The input is expected to be (B, 1, 1, 23, 23, 23)
        x = x.squeeze(1)
        if self.pe_mode == 'grid':
            x = self.pe(x)
        features = self.backbone(x)
        logits = self.head(features)
        
        # The loss function (like nn.CrossEntropyLoss) will handle softmax internally.
        # If you need log probabilities, apply them here.
        if self.head_mode == 'relu':
            logits = F.relu(logits) + 1e-10
            logits = logits / logits.sum(dim=1, keepdim=True)
            # log_probs = torch.log(logits)  # Convert to log probabilities
        elif self.head_mode == 'abs':
            logits = torch.abs(logits) + 1e-10
            logits = logits / logits.sum(dim=1, keepdim=True)
            # log_probs = torch.log(logits)  # Convert to log probabilities
        else:
            logits = F.softmax(logits, dim=1)
            # log_probs = F.log_softmax(logits, dim=1)

        return logits
        # return log_probs     
      

class FaceSelectorWeightUnified(nn.Module):
    """
    An efficient 3D face selector using a unified, depthwise-separable architecture.
    """
    def __init__(self, in_channels=1, name="FaceSelectorUnified_v1", head_mode="relu", pe_mode="grid"):
        super().__init__()
        self.name = name
        self.pe_mode = pe_mode
        self.pe   = PosEmbed3D()
        self.head_mode = head_mode
        if pe_mode == "grid":
            in_channels = in_channels + self.pe.pe.shape[1]
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
            nn.Flatten(),                    # -> (B, 64)
            nn.Linear(512, 256),         # -> (B, 6)
            nn.GELU(),
            nn.Linear(256, 128),         # -> (B, 6)
            nn.GELU(),
            nn.Linear(128, 7)         # -> (B, 6)
        )

    def forward(self, x):
        # The input is expected to be (B, 1, 1, 23, 23, 23)
        x = x.squeeze(1)
        if self.pe_mode == 'grid':
            x = self.pe(x)
        features = self.backbone(x)
        logits = self.head(features)

        if self.head_mode == 'relu':
            logits = F.relu(logits) + 1e-10
            # face_logits = torch.log(logits[:, :6] / logits[:, :6].sum(dim=1, keepdim=True))
            face_logits = logits[:, :6] / logits[:, :6].sum(dim=1, keepdim=True)
            logits = torch.cat([face_logits, logits[:, 6:]], dim=1)
        elif self.head_mode == 'abs':
            logits = torch.abs(logits) + 1e-10
            # face_logits = torch.log(logits[:, :6] / logits[:, :6].sum(dim=1, keepdim=True))
            face_logits = logits[:, :6] / logits[:, :6].sum(dim=1, keepdim=True)
            logits = torch.cat([face_logits, logits[:, 6:]], dim=1)
        else:
            face_logits = F.softmax(logits[:, :6], dim=1)
            # face_logits = F.log_softmax(logits[:, :6], dim=1)
            logits = torch.cat([face_logits, logits[:, 6:]], dim=1)
        return logits
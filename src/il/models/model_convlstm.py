# --- bootstrap de rutas: modulos de IL organizados por rol ---
import sys as _sys
from pathlib import Path as _Path
_IL_ROOT = _Path(__file__).resolve().parent.parent
for _sub in ('', 'models', 'data', 'train', 'serve', 'analysis'):
    _p = str(_IL_ROOT / _sub) if _sub else str(_IL_ROOT)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- fin bootstrap ---
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Celda ConvLSTM (Shi et al., 2015).

    Reemplaza las multiplicaciones matriciales de una LSTM estándar por
    convoluciones 2D, preservando la estructura espacial.

    Entrada por timestep : (B, input_channels, H, W)
    Estado oculto        : (h, c) cada uno (B, hidden_channels, H, W)
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2  # same padding

        # Una sola conv que produce las 4 gates (i, f, o, g) de golpe
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x, state):
        """
        Args:
            x     : (B, input_channels, H, W)
            state : (h, c) cada uno (B, hidden_channels, H, W)
        Returns:
            (h_next, c_next)
        """
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)

        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
        )


class CNNExtractor(nn.Module):
    """
    Extractor CNN que preserva dimensiones espaciales para ConvLSTM.

    Entrada : (N, 3, H, W)   — imágenes RGB
    Salida  : (N, out_channels, H', W') — feature maps espaciales

    Con IMG_SIZE=128: salida es (N, 64, 8, 8) tras 4 bloques de stride 2.
    """

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.out_channels = out_channels

        self.encoder = nn.Sequential(
            # 128x128 → 64x64
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 64x64 → 32x32
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            # 32x32 → 16x16
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 16x16 → 8x8
            nn.Conv2d(64, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, imgs):
        """(N, C, H, W) → (N, out_channels, H', W')"""
        return self.encoder(imgs)


class MinecraftConvLSTMModel(nn.Module):
    """
    Modelo temporal dual-head con ConvLSTM para IL.

    Pipeline:
      1. CNNExtractor:  (B*T, 3, H, W) → (B*T, C_feat, H', W')
      2. Reshape:       (B, T, C_feat, H', W')
      3. ConvLSTMCell:  procesa T steps → (B, hidden_ch, H', W')
      4. Global pool:   (B, hidden_ch)
      5. Fusión estado: concat con state_proj(state) → (B, hidden_ch + proj_dim)
      6. Dual heads:    action_head + camera_head
    """

    def __init__(self, num_actions, cnn_channels=64, hidden_channels=64,
                 state_dim=9, state_proj_dim=32, camera_dim=2):
        super().__init__()
        self.cnn_channels    = cnn_channels
        self.hidden_channels = hidden_channels
        self.state_dim       = state_dim
        self.camera_dim      = camera_dim

        self.extractor = CNNExtractor(out_channels=cnn_channels)

        self.convlstm = ConvLSTMCell(
            input_channels=cnn_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Proyección del estado
        self.state_proj = nn.Linear(state_dim, state_proj_dim)
        fused_dim = hidden_channels + state_proj_dim

        # Cabeza de clasificación (acción discreta)
        self.action_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fused_dim, num_actions),
        )

        # Cabeza de regresión (camera delta)
        self.camera_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fused_dim, camera_dim),
            nn.Tanh(),
        )

    def forward(self, imgs, states):
        """
        Args:
            imgs   : (B, T, C, H, W)
            states : (B, T, state_dim)
        Returns:
            logits      : (B, T, num_actions)
            camera_pred : (B, T, 2)
        """
        B, T, C, H, W = imgs.shape

        # Extraer feature maps espaciales
        feat_maps = self.extractor(imgs.reshape(B * T, C, H, W))  # (B*T, ch, H', W')
        _, ch, fH, fW = feat_maps.shape
        feat_maps = feat_maps.view(B, T, ch, fH, fW)

        # Procesar secuencia con ConvLSTM
        h, c = self.convlstm.init_hidden(B, fH, fW, imgs.device)

        all_logits = []
        all_camera = []

        for t in range(T):
            h, c = self.convlstm(feat_maps[:, t], (h, c))

            # Pool espacial → vector
            pooled = self.pool(h).view(B, -1)           # (B, hidden_channels)
            state_feat = self.state_proj(states[:, t])   # (B, state_proj_dim)
            fused = torch.cat([pooled, state_feat], dim=1)

            all_logits.append(self.action_head(fused))
            all_camera.append(self.camera_head(fused))

        logits      = torch.stack(all_logits, dim=1)  # (B, T, num_actions)
        camera_pred = torch.stack(all_camera, dim=1)  # (B, T, 2)
        return logits, camera_pred

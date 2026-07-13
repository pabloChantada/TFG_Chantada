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
import timm


class ViTExtractor(nn.Module):
    """
    Extractor de features visuales basado en Vision Transformer (ViT) via timm.

    Usa vit_small_patch16_224 con img_size=128 (patch 16×16 → 64 tokens).
    La cabeza de clasificación se reemplaza por una proyección lineal a feat_dim.

    Entrada : (N, C, H, W)  con H=W=img_size
    Salida  : (N, feat_dim)
    """

    def __init__(self, img_size: int = 128, feat_dim: int = 256,
                 model_name: str = "vit_small_patch16_224", pretrained: bool = True,
                 unfreeze_blocks: int = 0):
        """
        Args:
            unfreeze_blocks : nº de bloques Transformer finales que se entrenan.
                              0  → backbone completamente congelado (solo proj entrenable).
                             -1  → todo el backbone descongelado (fine-tuning completo).
        """
        super().__init__()
        self.feat_dim = feat_dim

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,      # elimina la cabeza de clasificación → salida embed_dim
        )
        vit_out_dim = self.vit.embed_dim   # 384 para vit_small
        self.proj = nn.Linear(vit_out_dim, feat_dim) if vit_out_dim != feat_dim else nn.Identity()

        self._apply_freeze(unfreeze_blocks)

    def _apply_freeze(self, unfreeze_blocks: int):
        # Congelar todo el backbone por defecto
        for p in self.vit.parameters():
            p.requires_grad = False

        if unfreeze_blocks == -1:
            # Descongelar todo
            for p in self.vit.parameters():
                p.requires_grad = True
        elif unfreeze_blocks > 0:
            # Descongelar los últimos N bloques + norm final
            blocks = list(self.vit.blocks)
            for block in blocks[-unfreeze_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.vit.norm.parameters():
                p.requires_grad = True

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs : (N, C, H, W)
        Returns:
            feats : (N, feat_dim)
        """
        return self.proj(self.vit(imgs))


class RNNExtractor(nn.Module):
    """
    Extractor de features visuales basado en GRU.
    Trata cada fila de la imagen como un paso temporal.

    Entrada : (N, C, H, W)  — N imágenes (el caller gestiona la dim. temporal)
    Salida  : (N, feat_dim)

    feat_dim = hidden_size * (2 si bidirectional, 1 si no)
    """

    def __init__(self, img_channels: int = 3, img_width: int = 128,
                 hidden_size: int = 512, num_layers: int = 2,
                 bidirectional: bool = False):
        super().__init__()
        self.feat_dim = hidden_size * (2 if bidirectional else 1)

        self.gru = nn.GRU(
            input_size=img_channels * img_width,   # cada fila aplanada: C*W
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs : (N, C, H, W)
        Returns:
            feats : (N, feat_dim)
        """
        N, C, H, W = imgs.shape
        # (N, C, H, W) -> (N, H, C*W)  — cada fila de la imagen es un timestep
        x = imgs.permute(0, 2, 1, 3).reshape(N, H, C * W)
        _, h_n = self.gru(x)   # h_n : (num_layers * D, N, hidden_size)
        return h_n[-1]         # última capa: (N, feat_dim)


class MinecraftILModel(nn.Module):
    """
    Modelo temporal dual-head para IL.

    Entrada:
      features : (B, T, feat_dim)  — deep features extraídas por RNNExtractor
      states   : (B, T, state_dim) — vector de estado normalizado (x,y,z,yaw,pitch,dx,dz,tree_visible,tree_distance)

    Arquitectura:
      1. El estado se proyecta a feat_dim con una capa lineal.
      2. Se suma a los deep features (no concatenación).
      3. Un LSTM procesa la secuencia fusionada.
      4. Dos cabezas paralelas sobre la salida LSTM:
         - action_head: clasificación → (B, T, num_actions)
         - camera_head: regresión    → (B, T, 2) con tanh → dyaw, dpitch en [-1, 1]

    Loss de entrenamiento : CrossEntropyLoss (acción) + MSELoss (cámara)
    Inferencia            : softmax(logits[:, -1, :]) para acción,
                            desnormalizar camera[:, -1, :] para dyaw/dpitch.
    """

    def __init__(self, num_actions: int, feat_dim: int = 512,
                 state_dim: int = 9, lstm_hidden: int = 256,
                 camera_dim: int = 2):
        super().__init__()
        self.feat_dim    = feat_dim
        self.state_dim   = state_dim
        self.lstm_hidden = lstm_hidden
        self.camera_dim  = camera_dim

        # Proyección del estado al espacio de deep features (para la suma)
        self.state_proj = nn.Linear(state_dim, feat_dim)

        # Procesado temporal
        self.lstm = nn.LSTM(feat_dim, lstm_hidden, batch_first=True)

        # Cabeza de clasificación (acción discreta)
        self.action_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden, num_actions),
        )

        # Cabeza de regresión (camera delta continuo)
        self.camera_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden, camera_dim),
            nn.Tanh(),  # salida en [-1, 1]
        )

    def forward(self, features, states: torch.Tensor):
        """
        Args:
            features : (B, T, feat_dim) o None (variante solo-estado, sin visión)
            states   : (B, T, state_dim)
        Returns:
            logits       : (B, T, num_actions)
            camera_pred  : (B, T, 2)  — dyaw, dpitch normalizados a [-1, 1]
        """
        proj = self.state_proj(states)               # (B, T, feat_dim)
        # En la variante solo-estado no hay características visuales: el tronco
        # opera únicamente sobre el estado proyectado.
        fused = proj if features is None else features + proj
        lstm_out, _ = self.lstm(fused)              # (B, T, lstm_hidden)
        logits      = self.action_head(lstm_out)    # (B, T, num_actions)
        camera_pred = self.camera_head(lstm_out)    # (B, T, 2)
        return logits, camera_pred

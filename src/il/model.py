import torch
import torch.nn as nn


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

    def forward(self, features: torch.Tensor, states: torch.Tensor):
        """
        Args:
            features : (B, T, feat_dim)
            states   : (B, T, state_dim)
        Returns:
            logits       : (B, T, num_actions)
            camera_pred  : (B, T, 2)  — dyaw, dpitch normalizados a [-1, 1]
        """
        fused = features + self.state_proj(states)  # (B, T, feat_dim)
        lstm_out, _ = self.lstm(fused)              # (B, T, lstm_hidden)
        logits      = self.action_head(lstm_out)    # (B, T, num_actions)
        camera_pred = self.camera_head(lstm_out)    # (B, T, 2)
        return logits, camera_pred

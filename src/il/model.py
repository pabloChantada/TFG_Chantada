import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

BACKBONES = {
    'resnet18':  (resnet18,  'IMAGENET1K_V1', 512),
    'resnet34':  (resnet34,  'IMAGENET1K_V1', 512),
    'resnet50':  (resnet50,  'IMAGENET1K_V2', 2048),
    'resnet101': (resnet101, 'IMAGENET1K_V2', 2048),
}


class ResNetExtractor(nn.Module):
    """
    Extractor de deep features visuales basado en ResNet preentrenado.

    Entrada : (N, C, H, W)  — N imágenes (el caller gestiona la dim. temporal)
    Salida  : (N, feat_dim)

    Capas congeladas: todo excepto layer4.
    """

    def __init__(self, backbone: str = 'resnet18'):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Backbone '{backbone}' no soportado. Opciones: {list(BACKBONES.keys())}")

        build_fn, weights, feat_dim = BACKBONES[backbone]
        base = build_fn(weights=weights)

        for name, param in base.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

        base.fc = nn.Identity()

        self.net           = base
        self.backbone_name = backbone
        self.feat_dim      = feat_dim

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.net(imgs)


class MinecraftILModel(nn.Module):
    """
    Modelo temporal para IL.

    Entrada:
      features : (B, T, feat_dim)  — deep features extraídas por ResNetExtractor
      states   : (B, T, state_dim) — vector de estado del bot (x, y, z, yaw, pitch)

    Arquitectura:
      1. El estado se proyecta a feat_dim con una capa lineal.
      2. Se suma a los deep features (no concatenación).
      3. Un LSTM procesa la secuencia fusionada.
      4. Una cabeza lineal produce logits por timestep: (B, T, num_actions)

    Loss de entrenamiento : CrossEntropyLoss (= CategoricalCrossEntropy, aplica
                            log-softmax internamente). El forward devuelve LOGITS.
    Inferencia            : aplicar F.softmax(logits[:, -1, :], dim=-1) sobre el
                            último timestep para obtener probabilidades.
    """

    def __init__(self, num_actions: int, feat_dim: int = 512,
                 state_dim: int = 5, lstm_hidden: int = 256):
        super().__init__()
        self.feat_dim    = feat_dim
        self.state_dim   = state_dim
        self.lstm_hidden = lstm_hidden

        # Proyección del estado al espacio de deep features (para la suma)
        self.state_proj = nn.Linear(state_dim, feat_dim)

        # Procesado temporal
        self.lstm = nn.LSTM(feat_dim, lstm_hidden, batch_first=True)

        # Cabeza de clasificación
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden, num_actions),
        )

    def forward(self, features: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features : (B, T, feat_dim)
            states   : (B, T, state_dim)
        Returns:
            logits   : (B, T, num_actions)
        """
        fused    = features + self.state_proj(states)  # (B, T, feat_dim)
        lstm_out, _ = self.lstm(fused)                 # (B, T, lstm_hidden)
        return self.head(lstm_out)                     # (B, T, num_actions)

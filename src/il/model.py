import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

# NUM_ACTIONS = número de clases atómicas del dataset
# (idle, move_forward=2, camera_yaw=2, attack=1, etc.)
# Actualmente tenemos 15, y sobreescribimos el valor en main.py
# pero lo dejamos aqui como default

BACKBONES = {
    'resnet18':  (resnet18,  'IMAGENET1K_V1'),
    'resnet34':  (resnet34,  'IMAGENET1K_V1'),
    'resnet50':  (resnet50,  'IMAGENET1K_V1'),
    'resnet101': (resnet101, 'IMAGENET1K_V1'),
}

class MinecraftILModel(nn.Module):
    def __init__(self, num_actions, backbone: str = 'resnet18'):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Backbone '{backbone}' no soportado. Opciones: {list(BACKBONES.keys())}")
        
        build_fn, weights = BACKBONES[backbone]
        base = build_fn(weights=weights)
        in_features = base.fc.in_features

        # Reemplazar fc final con la capa de acciones del dataset
        base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_actions)
        )

        self.model = base
        self.backbone_name = backbone

    def forward(self, x):
        # Entrada a la resnet: [B,3,224,224]
        logits = self.model(x)  # [B, num_actions] logits
        return logits
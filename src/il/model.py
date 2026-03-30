import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

BACKBONES = {
    'resnet18':  (resnet18,  'IMAGENET1K_V1'),
    'resnet34':  (resnet34,  'IMAGENET1K_V1'),
    'resnet50':  (resnet50,  'IMAGENET1K_V2'),
    'resnet101': (resnet101, 'IMAGENET1K_V2'),
}

class MinecraftILModel(nn.Module):
    def __init__(self, num_actions, backbone: str = 'resnet18', num_aux: int = 0):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Backbone '{backbone}' no soportado. Opciones: {list(BACKBONES.keys())}")

        build_fn, weights = BACKBONES[backbone]
        base = build_fn(weights=weights)
        in_features = base.fc.in_features

        # Congelar el layer 4
        for name, param in base.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

        # Replace fc with Identity to expose the pooled feature vector.
        # The actual head is defined below and receives visual + aux features.
        base.fc = nn.Identity()
    
        self.model = base
        self.backbone_name = backbone
        self.num_aux = num_aux

        # Head: concatenates visual features (in_features) + aux scalars (num_aux)
        self.head = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(in_features + num_aux, num_actions)
        )

    def forward(self, x, aux=None):
        feats = self.model(x)  # (B, in_features)
        if self.num_aux > 0:
            if aux is not None:
                feats = torch.cat([feats, aux], dim=1)
            else:
                # Inference without state: fill with zeros (graceful degradation)
                zeros = torch.zeros(feats.size(0), self.num_aux, device=feats.device)
                feats = torch.cat([feats, zeros], dim=1)
        return self.head(feats)

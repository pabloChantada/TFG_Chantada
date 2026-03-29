import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

BACKBONES = {
    'resnet18':  (resnet18,  'IMAGENET1K_V1'),
    'resnet34':  (resnet34,  'IMAGENET1K_V1'),
    'resnet50':  (resnet50,  'IMAGENET1K_V2'),
    'resnet101': (resnet101, 'IMAGENET1K_V2'),
}

class MinecraftILModel(nn.Module):
    def __init__(self, num_actions, backbone: str = 'resnet18'):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Backbone '{backbone}' no soportado. Opciones: {list(BACKBONES.keys())}")

        build_fn, weights = BACKBONES[backbone]
        base = build_fn(weights=weights)
        in_features = base.fc.in_features

        # Congelar todo excepto layer4 y fc.
        # Con ~4k ejemplos, entrenar el backbone completo causa overfitting severo.
        # layer4 (~2.5M params) captura features de alto nivel específicas de la tarea;
        # layers 1-3 (~8.5M params) ya tienen features genéricas útiles de ImageNet.
        for name, param in base.named_parameters():
            if not name.startswith("layer4") and not name.startswith("fc"):
                param.requires_grad = False

        base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_actions)
        )

        self.model = base
        self.backbone_name = backbone

    def forward(self, x):
        return self.model(x)

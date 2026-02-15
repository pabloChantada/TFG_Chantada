"""VAE para comprimir screenshots de Minecraft.

Objetivo:
1) Probar diferentes arquitecturas.
2) Compararlas.
"""

'''
Probar l2 vs l1
Probar diferentes tamaños del VAE en capas 3, 4, 5
Probar con y sin lr scheduler
'''

import os
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from vae_visuals import plot_metrics_bar, plot_reconstructions, plot_training_curves


# --- CONFIGURACION ---
AGENT_METRICS_PATH = "/Users/circus/repos/TFG_Chantada/src/metrics/agent_metrics"
LEARNING_RATE = 1e-3
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)

VAE_CONFIGS = {
    "small": {"image_size": 64, "latent_dim": 32, "batch_size": 128, "epochs": 10},
    "medium": {"image_size": 128, "latent_dim": 64, "batch_size": 128, "epochs": 10},
    "large": {"image_size": 128, "latent_dim": 128, "batch_size": 128, "epochs": 10},
}


class MinecraftScreenshotsDataset(Dataset):
    def __init__(self, root_dir, augment=True, image_size=128):
        self.image_paths = []

        if os.path.basename(root_dir) == "agent_metrics" or "agent_metrics" in root_dir:
            screenshot_dirs = glob.glob(os.path.join(root_dir, "*_screenshots"))
            print(f"Encontrados {len(screenshot_dirs)} directorios de screenshots")
            for screenshot_dir in screenshot_dirs:
                images = glob.glob(os.path.join(screenshot_dir, "*.jpg")) + glob.glob(
                    os.path.join(screenshot_dir, "*.png")
                )
                self.image_paths.extend(images)
                print(f"  - {os.path.basename(screenshot_dir)}: {len(images)} imagenes")
        else:
            self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg")) + glob.glob(
                os.path.join(root_dir, "*.png")
            )

        if not self.image_paths:
            raise RuntimeError(f"No se encontraron imagenes en {root_dir}")

        print(f"\nTotal: {len(self.image_paths)} imagenes para entrenamiento.")
        self.image_size = image_size

        base_transform = [transforms.Resize((self.image_size, self.image_size))]
        if augment:
            augment_transform = [
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
            transform_list = augment_transform + base_transform + [transforms.ToTensor()]
        else:
            transform_list = base_transform + [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            return self.transform(image)
        except Exception as exc:
            print(f"Error cargando imagen {img_path}: {exc}")
            return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)


class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=64, image_size=128):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim

        import math

        num_layers = int(math.log2(image_size))

        encoder_layers = []
        in_channels = img_channels
        out_channels = 16

        for _ in range(num_layers):
            encoder_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                ]
            )
            in_channels = out_channels
            out_channels = min(out_channels * 2, 1024)

        self.encoder = nn.Sequential(*encoder_layers)
        self.final_channels = in_channels

        self.fc_mu = nn.Linear(self.final_channels * 1 * 1, latent_dim)
        self.fc_logvar = nn.Linear(self.final_channels * 1 * 1, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.final_channels * 1 * 1)

        decoder_layers = []
        in_channels = self.final_channels
        for _ in range(num_layers - 1):
            out_channels = in_channels // 2
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                ]
            )
            in_channels = out_channels

        decoder_layers.extend(
            [nn.ConvTranspose2d(in_channels, img_channels, kernel_size=4, stride=2, padding=1), nn.Sigmoid()]
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), self.final_channels, 1, 1)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    reconstruction_loss = F.l1_loss(recon_x, x, reduction="sum")
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + (beta * kld_loss), reconstruction_loss, kld_loss


def train(screenshot_path, image_size, latent_dim, batch_size, epochs, save_path, verbose=True, log_interval=2):
    full_path = os.path.abspath(screenshot_path)
    if not os.path.exists(full_path):
        full_path = os.path.abspath(os.path.join(os.getcwd(), screenshot_path))

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO VAE")
    print("=" * 60)
    print(f"Dispositivo: {DEVICE}")
    print(f"Dataset: {full_path}")
    print(f"Config: image_size={image_size}, latent_dim={latent_dim}, batch_size={batch_size}, epochs={epochs}")
    print("=" * 60 + "\n")

    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)

    checkpoint_name = f"best_weights_img{image_size}_latent{latent_dim}_lr{LEARNING_RATE:.0e}_bs{batch_size}.pth"
    checkpoint_path = os.path.join(weights_dir, checkpoint_name)

    dataset = MinecraftScreenshotsDataset(full_path, augment=True, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VAE(latent_dim=latent_dim, image_size=image_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    kl_annealing_epochs = 20
    best_loss = float("inf")

    history = {
        "avg_loss": [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        beta = min(1.0, epoch / kl_annealing_epochs)

        for batch_idx, data in enumerate(dataloader):
            data = data.to(DEVICE)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss, _, _ = loss_function(recon_batch, data, mu, logvar, beta)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if verbose and batch_idx % log_interval == 0:
                print(
                    f"Epoch {epoch+1} [{(batch_idx+1)*len(data)}/{len(dataset)}] "
                    f"Loss: {loss.item() / len(data):.4f}"
                )

        avg_loss = train_loss / len(dataset)
        history["avg_loss"].append(avg_loss)
        print(f"====> Epoch {epoch+1} Average loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"   Nuevo mejor loss. Guardado en {checkpoint_path}")

        scheduler.step()

    save_dict = {
        "model_state_dict": model.state_dict(),
        "image_size": image_size,
        "latent_dim": latent_dim,
        "num_images": len(dataset),
        "epochs": epochs,
    }
    torch.save(save_dict, save_path)
    print(f"\nModelo guardado como '{save_path}'")
    print(f"- Imagenes entrenadas: {len(dataset)}")
    print(f"- Epocas: {epochs}")
    print(f"- Latent dim: {latent_dim}")

    return model, history


def predict(model, image_path, image_size=None):
    model.eval()
    if image_size is None:
        image_size = getattr(model, "image_size", 128)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mu, logvar = model.encode(image_tensor)
        z = model.reparameterize(mu, logvar)
        recon = model.decode(z)

    return {
        "reconstruction": recon.cpu().numpy(),
        "original": image_tensor.cpu().numpy(),
    }


def evaluate_model(model, test_images, model_name="VAE"):
    model.eval()
    total_mse = 0
    total_mae = 0

    print(f"\nEvaluando {model_name} en {len(test_images)} imagenes...")

    with torch.no_grad():
        for img_path in test_images:
            result = predict(model, img_path)
            original = torch.from_numpy(result["original"])
            recon = torch.from_numpy(result["reconstruction"])

            total_mse += F.mse_loss(recon, original).item()
            total_mae += F.l1_loss(recon, original).item()

    metrics = {
        "model_name": model_name,
        "num_images": len(test_images),
        "avg_mse": total_mse / len(test_images),
        "avg_mae": total_mae / len(test_images),
    }

    print(f"Resultados {model_name}:")
    print(f"- MSE promedio: {metrics['avg_mse']:.6f}")
    print(f"- MAE promedio: {metrics['avg_mae']:.6f}")

    return metrics


def compare_architectures(base_path=AGENT_METRICS_PATH, sample_images=10, seed=7):
    image_files = []
    screenshot_dirs = glob.glob(os.path.join(base_path, "*_screenshots"))
    for screenshot_dir in screenshot_dirs:
        image_files.extend(glob.glob(os.path.join(screenshot_dir, "*.jpg")))
        image_files.extend(glob.glob(os.path.join(screenshot_dir, "*.png")))

    if not image_files:
        raise RuntimeError(f"No se encontraron imagenes en {base_path}")

    random.seed(seed)
    test_images = random.sample(image_files, min(sample_images, len(image_files)))

    results = []
    histories = {}
    trained_models = {}
    for name, cfg in VAE_CONFIGS.items():
        print("\n" + "=" * 80)
        print(f"Entrenando arquitectura: {name}")
        print("=" * 80)

        save_path = f"vae_minecraft_{name}.pth"
        model, history = train(
            screenshot_path=base_path,
            image_size=cfg["image_size"],
            latent_dim=cfg["latent_dim"],
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            save_path=save_path,
            verbose=True,
            log_interval=2,
        )

        metrics = evaluate_model(model, test_images, model_name=name)
        results.append(metrics)
        histories[name] = history
        trained_models[name] = model

    results.sort(key=lambda x: x["avg_mse"])
    print("\n" + "=" * 80)
    print("Ranking final (menor MSE es mejor):")
    for idx, item in enumerate(results, start=1):
        print(f"{idx}. {item['model_name']} | MSE {item['avg_mse']:.6f} | MAE {item['avg_mae']:.6f}")
    print("=" * 80)

    plot_training_curves(histories)
    plot_metrics_bar(results)
    plot_reconstructions(trained_models, test_images, predict)


if __name__ == "__main__":
    compare_architectures()
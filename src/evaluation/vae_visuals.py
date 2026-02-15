"""Plot helpers for VAE comparisons."""

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

def plot_training_curves(histories, save_path="vae_training_curves.png"):
    if not histories:
        return

    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        plt.plot(history["avg_loss"], label=name)

    plt.title("Comparacion de loss por epoca")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curvas de entrenamiento guardadas en {save_path}")


def plot_metrics_bar(results, save_path="vae_metrics_comparison.png"):
    if not results:
        return

    names = [item["model_name"] for item in results]
    mse_vals = [item["avg_mse"] for item in results]
    mae_vals = [item["avg_mae"] for item in results]

    x = range(len(names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], mse_vals, width=width, label="MSE")
    plt.bar([i + width / 2 for i in x], mae_vals, width=width, label="MAE")
    plt.xticks(list(x), names)
    plt.title("Comparacion de MSE y MAE")
    plt.xlabel("Modelo")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparacion de metricas guardada en {save_path}")


def plot_reconstructions(models, test_images, predict_fn, save_path="vae_recon_comparison.png", max_images=3):
    if not models or not test_images:
        return

    images = test_images[:max_images]
    num_models = len(models)
    num_images = len(images)

    fig, axes = plt.subplots(num_images, num_models + 1, figsize=(4 * (num_models + 1), 4 * num_images))
    if num_images == 1:
        axes = [axes]

    for img_idx, img_path in enumerate(images):
        image = Image.open(img_path).convert("RGB")
        axes[img_idx][0].imshow(image)
        axes[img_idx][0].set_title("Original")
        axes[img_idx][0].axis("off")

        for model_idx, (name, model) in enumerate(models.items()):
            result = predict_fn(model, img_path)
            reconstruction = result["reconstruction"][0].transpose(1, 2, 0)
            axes[img_idx][model_idx + 1].imshow(reconstruction)
            axes[img_idx][model_idx + 1].set_title(name)
            axes[img_idx][model_idx + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparacion de reconstrucciones guardada en {save_path}")


def _default_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )


def _predict_default(model, image_path, device):
    model.eval()
    image_size = getattr(model, "image_size", 128)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        mu, logvar = model.encode(image_tensor)
        z = model.reparameterize(mu, logvar)
        recon = model.decode(z)

    return {
        "reconstruction": recon.cpu().numpy(),
        "original": image_tensor.cpu().numpy(),
    }


def _load_vae_from_checkpoint(model_path, device):
    from vae_train import VAE

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        latent_dim = checkpoint.get("latent_dim", 64)
        image_size = checkpoint.get("image_size", 128)
        model = VAE(latent_dim=latent_dim, image_size=image_size).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = VAE(latent_dim=64, image_size=128).to(device)
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def plot_reconstructions_from_paths(
    model_paths,
    test_images,
    save_path="vae_recon_comparison.png",
    max_images=3,
    device=None,
):
    if not model_paths:
        return

    device = device or _default_device()
    models = {}
    for path in model_paths:
        name = path.replace(".pth", "").split("/")[-1]
        models[name] = _load_vae_from_checkpoint(path, device)

    plot_reconstructions(models, test_images, lambda m, p: _predict_default(m, p, device), save_path, max_images)

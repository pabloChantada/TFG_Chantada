"""
VAE (Variational Autoencoder) para comprimir screenshots de Minecraft
Entrena un modelo para comprimir imágenes 64x64 a un vector latente de 32 dimensiones
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
import random


# --- CONFIGURACIÓN ---
SCREENSHOT_PATH = "/Users/circus/repos/TFG_Chantada/src/metrics/agent_metrics/Agent1/screenshots"
IMAGE_SIZE = 128       # Aumentado a 128x128 para más detalle
LATENT_DIM = 128       # Tamaño del vector z comprimido
BATCH_SIZE = 16        # Reducido para 128x128 (usa más memoria)
LEARNING_RATE = 1e-4
EPOCHS = 100
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")


# --- 1. DATASET ---
class MinecraftScreenshotsDataset(Dataset):
    def __init__(self, root_dir, augment=True):
        # Busca todas las imágenes jpg/png en la carpeta
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg")) + \
                           glob.glob(os.path.join(root_dir, "*.png"))
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No se encontraron imágenes en {root_dir}")
            
        print(f"Cargadas {len(self.image_paths)} imágenes para entrenamiento.")

        # Transformaciones base (siempre aplica)
        base_transform = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ]
        
        # Transformaciones de augmentación (solo durante entrenamiento)
        if augment:
            augment_transform = [
                # Toma un trozo aleatorio de la imagen y lo estira a 64x64.
                # Scale=(0.8, 1.0) significa que hace zoom entre un 0% y un 20%.
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)), 
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
            image = Image.open(img_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error cargando imagen {img_path}: {e}")
            # Return a zero tensor as a tensor (already transformed)
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)


# --- 2. MODELO VAE ---
class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=64):
        super(VAE, self).__init__()
        
        # --- ENCODER MEJORADO ---
        self.encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(img_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            # 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 4x4 -> 2x2
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 2x2 -> 1x1 (capa extra para 128x128)
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )
        
        self.fc_mu = nn.Linear(1024 * 1 * 1, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 1 * 1, latent_dim)

        # --- DECODER MEJORADO ---
        self.decoder_input = nn.Linear(latent_dim, 1024 * 1 * 1)
        
        self.decoder = nn.Sequential(
            # 1x1 -> 2x2 (capa extra para 128x128)
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 2x2 -> 4x4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(16, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Salida final entre 0 y 1
        )

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
        h = h.view(h.size(0), 1024, 1, 1)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    


# --- 3. LOSS FUNCTION ---
# Modifica tu loss function
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Usar L1 en lugar de MSE para bordes más nítidos en Minecraft
    reconstruction_loss = F.l1_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Beta controla el peso de la regularización (concepto Beta-VAE)
    return reconstruction_loss + (beta * kld_loss), reconstruction_loss, kld_loss


# --- 4. BUCLE DE ENTRENAMIENTO ---
def train(screenshot_path=SCREENSHOT_PATH, epochs=EPOCHS, save_path="vae_minecraft.pth"):
    """
    Entrena el VAE con las imágenes del directorio especificado
    
    Args:
        screenshot_path: Ruta al directorio con las imágenes
        epochs: Número de épocas de entrenamiento
        save_path: Ruta donde guardar el modelo entrenado
    
    Returns:
        model: Modelo VAE entrenado
    """
    # Ajustar ruta relativa
    full_path = os.path.abspath(screenshot_path)
    if not os.path.exists(full_path):
        full_path = os.path.abspath(os.path.join(os.getcwd(), screenshot_path))
    
    print(f"Usando dispositivo: {DEVICE}")
    print(f"Buscando dataset en: {full_path}")
    
    # Dataset con data augmentation
    dataset = MinecraftScreenshotsDataset(full_path, augment=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Modelo
    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    model.train()
    # Dentro de tu función train...
    NUM_EPOCHS = 10  # Aumenta las épocas, 20 es muy poco para un VAE desde cero
    kl_annealing_epochs = 20 # Tardará 20 épocas en aplicar todo el peso KL

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        
        # Calcular beta para esta época (de 0.0 a 1.0 gradualmente)
        # Esto permite al encoder aprender a reconstruir primero, y luego organizar el espacio latente.
        beta = min(1.0, epoch / kl_annealing_epochs)
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            
            # Pasamos beta a la loss function
            loss, rec_loss, kld_loss = loss_function(recon_batch, data, mu, logvar, beta)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 2 == 0:
                print(f'Epoch {epoch+1} [{(batch_idx+1)*len(data)}/{len(dataset)}] '
                      f'Loss: {loss.item() / len(data):.4f}')
                with open("vae_training_log.txt", "a") as log_file:
                    log_file.write(f'Epoch {epoch+1} Batch {batch_idx+1} '
                                   f'Loss: {loss.item() / len(data):.4f} '
                                   f'Rec: {rec_loss.item() / len(data):.4f} '
                                   f'KLD: {kld_loss.item() / len(data):.4f} '
                                   f'Beta: {beta:.4f}\n')

        avg_loss = train_loss / len(dataset)
        print(f'====> Epoch {epoch+1} Average loss: {avg_loss:.4f}')

    # Guardar modelo
    torch.save(model.state_dict(), save_path)
    print(f"Modelo guardado como '{save_path}'")
    return model


# --- 5. CARGA Y PREDICCIÓN ---
def load_model(model_path="vae_minecraft.pth"):
    """Carga un modelo VAE desde disco"""
    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def predict(model, image_path):
    """
    Procesa una imagen y obtiene su código latente
    
    Args:
        model: Modelo VAE
        image_path: Ruta a la imagen
    
    Returns:
        dict con 'latent_vector', 'reconstruction', 'original'
    """
    # Poner el modelo en modo evaluación (desactiva BatchNorm, Dropout, etc.)
    model.eval()
    
    # Sin augmentación para predicción
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        mu, logvar = model.encode(image_tensor)
        z = model.reparameterize(mu, logvar)
        recon = model.decode(z)
    
    return {
        'latent_vector': z.cpu().numpy(),
        'reconstruction': recon.cpu().numpy(),
        'original': image_tensor.cpu().numpy()
    }


# --- 6. VISUALIZACIÓN ---
def visualize_prediction(result):
    """Visualiza la imagen original, la reconstrucción y el código latente"""
    original = result['original'][0].transpose(1, 2, 0)
    reconstruction = result['reconstruction'][0].transpose(1, 2, 0)
    latent = result['latent_vector'][0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title('Imagen Original (128x128)')
    axes[0].axis('off')
    
    # Reconstrucción
    axes[1].imshow(reconstruction)
    axes[1].set_title('Reconstrucción VAE')
    axes[1].axis('off')
    
    # Código latente como heatmap
    latent_2d = latent.reshape(8, 16)
    im = axes[2].imshow(latent_2d, cmap='viridis')
    axes[2].set_title('Código Latente (128 dims -> 8x16)')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.show()


# --- MAIN ---
if __name__ == "__main__":
    # Entrenar el modelo
    print("Iniciando entrenamiento del VAE...")
    import time
    start_time = time.time()
    # model = train()
    end_time = time.time()
    print(f"Tiempo total de entrenamiento: {(end_time - start_time)/60:.2f} minutos")
    model = load_model("vae_minecraft.pth")  # Cargar modelo ya entrenado para pruebas rápidas
    print("\nEntrenamiento completado!")
    print("\nPara usar el modelo:")
    print("  model = load_model('vae_minecraft.pth')")
    print("  result = predict(model, 'ruta/a/imagen.png')")
    print("  visualize_prediction(result)")

    prediction_example = "src/metrics/example.png"
    prediction_result = predict(model, prediction_example)
    visualize_prediction(prediction_result)
    # Obtener un archivo aleatorio del directorio
    screenshots_dir = "src/metrics/agent_metrics/Agent1/screenshots"
    image_files = glob.glob(os.path.join(screenshots_dir, "*.png")) + \
                  glob.glob(os.path.join(screenshots_dir, "*.jpg"))
    
    if image_files:
        for _ in range(3):  # Probar con 3 imágenes aleatorias
            random_image = random.choice(image_files)
            print(f"\nProbando con imagen: {random_image}")
            result = predict(model, random_image)
            visualize_prediction(result)
    else:
        print(f"No se encontraron imágenes en {screenshots_dir}")

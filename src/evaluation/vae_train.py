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
# Busca screenshots de TODOS los agentes en agent_metrics
AGENT_METRICS_PATH = "/Users/circus/repos/TFG_Chantada/src/metrics/agent_metrics"
IMAGE_SIZE = 128       # Aumentado a 128x128 para más detalle
LATENT_DIM = 128       # Tamaño del vector z comprimido
BATCH_SIZE = 16        # Reducido para 128x128 (usa más memoria)
LEARNING_RATE = 1e-4
EPOCHS = 100
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

# Configuraciones de VAE para experimentar
VAE_CONFIGS = {
    "small": {"image_size": 64, "latent_dim": 32, "batch_size": 32, "epochs": 50},
    "medium": {"image_size": 128, "latent_dim": 128, "batch_size": 16, "epochs": 100},
    "large": {"image_size": 128, "latent_dim": 256, "batch_size": 8, "epochs": 100},
    "deep": {"image_size": 128, "latent_dim": 128, "batch_size": 16, "epochs": 150},
}


# --- 1. DATASET ---
class MinecraftScreenshotsDataset(Dataset):
    def __init__(self, root_dir, augment=True, image_size=IMAGE_SIZE):
        # Busca todas las imágenes jpg/png recursivamente en todas las carpetas de screenshots
        self.image_paths = []
        
        # Si root_dir es agent_metrics, busca en todas las subcarpetas *_screenshots
        if os.path.basename(root_dir) == "agent_metrics" or "agent_metrics" in root_dir:
            screenshot_dirs = glob.glob(os.path.join(root_dir, "*_screenshots"))
            print(f"Encontrados {len(screenshot_dirs)} directorios de screenshots")
            for screenshot_dir in screenshot_dirs:
                images = glob.glob(os.path.join(screenshot_dir, "*.jpg")) + \
                         glob.glob(os.path.join(screenshot_dir, "*.png"))
                self.image_paths.extend(images)
                print(f"  - {os.path.basename(screenshot_dir)}: {len(images)} imágenes")
        else:
            # Compatibilidad con ruta directa a un directorio específico
            self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg")) + \
                               glob.glob(os.path.join(root_dir, "*.png"))
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No se encontraron imágenes en {root_dir}")
            
        print(f"\n📊 Total: {len(self.image_paths)} imágenes para entrenamiento.")
        self.image_size = image_size

        # Transformaciones base (siempre aplica)
        base_transform = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        
        # Transformaciones de augmentación (solo durante entrenamiento)
        if augment:
            augment_transform = [
                # Toma un trozo aleatorio de la imagen y lo estira al tamaño deseado.
                # Scale=(0.8, 1.0) significa que hace zoom entre un 0% y un 20%.
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
            image = Image.open(img_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error cargando imagen {img_path}: {e}")
            # Return a zero tensor as a tensor (already transformed)
            return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)


# --- 2. MODELO VAE ADAPTATIVO ---
class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=64, image_size=128):
        super(VAE, self).__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Calcular número de capas necesarias
        # Para 64x64: log2(64) = 6 capas
        # Para 128x128: log2(128) = 7 capas
        import math
        num_layers = int(math.log2(image_size))
        
        # Construir encoder dinámicamente
        encoder_layers = []
        in_channels = img_channels
        out_channels = 16
        
        for i in range(num_layers):
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ])
            in_channels = out_channels
            out_channels = min(out_channels * 2, 1024)  # Cap at 1024 channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.final_channels = in_channels
        
        # Fully connected layers para mu y logvar
        self.fc_mu = nn.Linear(self.final_channels * 1 * 1, latent_dim)
        self.fc_logvar = nn.Linear(self.final_channels * 1 * 1, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.final_channels * 1 * 1)
        
        # Construir decoder dinámicamente (inverso del encoder)
        decoder_layers = []
        in_channels = self.final_channels
        
        for i in range(num_layers - 1):
            out_channels = in_channels // 2
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ])
            in_channels = out_channels
        
        # Última capa sin BatchNorm ni LeakyReLU
        decoder_layers.extend([
            nn.ConvTranspose2d(in_channels, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        ])
        
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
def train(screenshot_path=AGENT_METRICS_PATH, epochs=EPOCHS, save_path="vae_minecraft.pth", 
          config=None, image_size=IMAGE_SIZE, latent_dim=LATENT_DIM, batch_size=BATCH_SIZE):
    """
    Entrena el VAE con las imágenes del directorio especificado
    
    Args:
        screenshot_path: Ruta al directorio con las imágenes (por defecto carga de todos los agentes)
        epochs: Número de épocas de entrenamiento
        save_path: Ruta donde guardar el modelo entrenado
        config: Configuración predefinida ("small", "medium", "large", "deep")
        image_size: Tamaño de las imágenes
        latent_dim: Dimensión del espacio latente
        batch_size: Tamaño del batch
    
    Returns:
        model: Modelo VAE entrenado
    """
    # Si se proporciona una configuración, usarla
    if config and config in VAE_CONFIGS:
        cfg = VAE_CONFIGS[config]
        image_size = cfg["image_size"]
        latent_dim = cfg["latent_dim"]
        batch_size = cfg["batch_size"]
        epochs = cfg["epochs"]
        print(f"\n🎯 Usando configuración '{config}': {cfg}")
    # Ajustar ruta relativa
    full_path = os.path.abspath(screenshot_path)
    if not os.path.exists(full_path):
        full_path = os.path.abspath(os.path.join(os.getcwd(), screenshot_path))
    
    print(f"\n{'='*60}")
    print(f"🚀 ENTRENAMIENTO VAE")
    print(f"{'='*60}")
    print(f"Dispositivo: {DEVICE}")
    print(f"Buscando dataset en: {full_path}")
    print(f"Configuración: image_size={image_size}, latent_dim={latent_dim}, batch_size={batch_size}, epochs={epochs}")
    print(f"{'='*60}\n")
    
    # Dataset con data augmentation
    dataset = MinecraftScreenshotsDataset(full_path, augment=True, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Modelo
    model = VAE(latent_dim=latent_dim, image_size=image_size).to(DEVICE)
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

    # Guardar modelo con metadatos
    save_dict = {
        'model_state_dict': model.state_dict(),
        'image_size': image_size,
        'latent_dim': latent_dim,
        'num_images': len(dataset),
        'epochs': NUM_EPOCHS,
    }
    torch.save(save_dict, save_path)
    print(f"\n✅ Modelo guardado como '{save_path}'")
    print(f"   - Imágenes entrenadas: {len(dataset)}")
    print(f"   - Épocas: {NUM_EPOCHS}")
    print(f"   - Latent dim: {latent_dim}")
    return model


# --- 5. CARGA Y PREDICCIÓN ---
def load_model(model_path="vae_minecraft.pth"):
    """Carga un modelo VAE desde disco"""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Soporte para modelos antiguos sin metadatos
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        latent_dim = checkpoint.get('latent_dim', LATENT_DIM)
        image_size = checkpoint.get('image_size', IMAGE_SIZE)
        model = VAE(latent_dim=latent_dim, image_size=image_size).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modelo cargado: latent_dim={latent_dim}, image_size={image_size}")
    else:
        # Modelo antiguo
        model = VAE(latent_dim=LATENT_DIM, image_size=IMAGE_SIZE).to(DEVICE)
        model.load_state_dict(checkpoint)
        print(f"Modelo antiguo cargado con latent_dim={LATENT_DIM}, image_size={IMAGE_SIZE}")
    
    model.eval()
    return model


def predict(model, image_path, image_size=None):
    """
    Procesa una imagen y obtiene su código latente
    
    Args:
        model: Modelo VAE
        image_path: Ruta a la imagen
        image_size: Tamaño esperado de la imagen (None para usar el del modelo)
    
    Returns:
        dict con 'latent_vector', 'reconstruction', 'original'
    """
    # Poner el modelo en modo evaluación (desactiva BatchNorm, Dropout, etc.)
    model.eval()
    
    # Usar el image_size del modelo si no se proporciona
    if image_size is None:
        image_size = getattr(model, 'image_size', IMAGE_SIZE)
    
    # Sin augmentación para predicción
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
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
def visualize_prediction(result, title="VAE Prediction"):
    """Visualiza la imagen original, la reconstrucción y el código latente"""
    original = result['original'][0].transpose(1, 2, 0)
    reconstruction = result['reconstruction'][0].transpose(1, 2, 0)
    latent = result['latent_vector'][0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title(f'Imagen Original ({original.shape[0]}x{original.shape[1]})')
    axes[0].axis('off')
    
    # Reconstrucción
    axes[1].imshow(reconstruction)
    axes[1].set_title('Reconstrucción VAE')
    axes[1].axis('off')
    
    # Código latente como heatmap
    latent_dim = len(latent)
    # Crear una disposición cuadrada o rectangular para visualizar
    rows = int(np.sqrt(latent_dim))
    cols = int(np.ceil(latent_dim / rows))
    latent_2d = np.pad(latent, (0, rows*cols - latent_dim)).reshape(rows, cols)
    
    im = axes[2].imshow(latent_2d, cmap='viridis')
    axes[2].set_title(f'Código Latente ({latent_dim} dims -> {rows}x{cols})')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.show()


# --- 7. COMPARACIÓN DE MODELOS ---
def compare_models(models_dict, test_images, save_path="vae_comparison.png"):
    """
    Compara múltiples modelos VAE en las mismas imágenes de prueba
    
    Args:
        models_dict: Dict con {nombre: model_path}
        test_images: Lista de rutas a imágenes de prueba
        save_path: Donde guardar la comparación
    """
    num_models = len(models_dict)
    num_images = len(test_images)
    
    fig, axes = plt.subplots(num_images, num_models + 1, figsize=(5*(num_models+1), 5*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    # Cargar todos los modelos
    models = {}
    for name, path in models_dict.items():
        print(f"Cargando modelo {name}...")
        models[name] = load_model(path)
    
    # Para cada imagen de prueba
    for img_idx, img_path in enumerate(test_images):
        print(f"\nProcesando imagen {img_idx+1}/{num_images}: {img_path}")
        
        # Mostrar original
        image = Image.open(img_path).convert('RGB')
        axes[img_idx, 0].imshow(image)
        axes[img_idx, 0].set_title('Original')
        axes[img_idx, 0].axis('off')
        
        # Mostrar reconstrucción de cada modelo
        for model_idx, (name, model) in enumerate(models.items()):
            result = predict(model, img_path)
            reconstruction = result['reconstruction'][0].transpose(1, 2, 0)
            
            axes[img_idx, model_idx + 1].imshow(reconstruction)
            axes[img_idx, model_idx + 1].set_title(f'{name}')
            axes[img_idx, model_idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparación guardada en {save_path}")
    plt.show()


def evaluate_model(model, test_images, model_name="VAE"):
    """
    Evalúa un modelo VAE en un conjunto de imágenes
    
    Args:
        model: Modelo VAE
        test_images: Lista de rutas a imágenes de prueba
        model_name: Nombre del modelo para el reporte
    
    Returns:
        dict con métricas de evaluación
    """
    model.eval()
    
    total_recon_loss = 0
    total_mse = 0
    total_mae = 0
    
    print(f"\n📊 Evaluando {model_name} en {len(test_images)} imágenes...")
    
    with torch.no_grad():
        for img_path in test_images:
            result = predict(model, img_path)
            original = torch.from_numpy(result['original'])
            recon = torch.from_numpy(result['reconstruction'])
            
            # Calcular métricas
            mse = F.mse_loss(recon, original).item()
            mae = F.l1_loss(recon, original).item()
            
            total_mse += mse
            total_mae += mae
    
    metrics = {
        'model_name': model_name,
        'num_images': len(test_images),
        'avg_mse': total_mse / len(test_images),
        'avg_mae': total_mae / len(test_images),
    }
    
    print(f"✅ Resultados {model_name}:")
    print(f"   - MSE promedio: {metrics['avg_mse']:.6f}")
    print(f"   - MAE promedio: {metrics['avg_mae']:.6f}")
    
    return metrics


def train_all_configs(base_path=AGENT_METRICS_PATH):
    """
    Entrena modelos con todas las configuraciones predefinidas
    
    Args:
        base_path: Ruta al directorio con las imágenes
    
    Returns:
        dict con los paths de los modelos entrenados
    """
    models_trained = {}
    
    print(f"\n{'='*80}")
    print(f"🚀 ENTRENAMIENTO DE MÚLTIPLES CONFIGURACIONES VAE")
    print(f"{'='*80}\n")
    
    for config_name in VAE_CONFIGS.keys():
        print(f"\n{'='*80}")
        print(f"📦 Entrenando configuración: {config_name}")
        print(f"{'='*80}")
        
        save_path = f"vae_minecraft_{config_name}.pth"
        
        try:
            model = train(
                screenshot_path=base_path,
                config=config_name,
                save_path=save_path
            )
            models_trained[config_name] = save_path
            print(f"✅ {config_name} completado!")
        except Exception as e:
            print(f"❌ Error entrenando {config_name}: {e}")
    
    print(f"\n{'='*80}")
    print(f"🎉 ENTRENAMIENTO COMPLETADO")
    print(f"{'='*80}")
    print(f"Modelos entrenados: {list(models_trained.keys())}")
    
    return models_trained


# --- MAIN ---
if __name__ == "__main__":
    # ========================================
    # MODO 1: Entrenar una configuración específica
    # ========================================
    # print("\n" + "="*80)
    # print("🎯 MODO: Entrenar configuración específica")
    # print("="*80)
    
    # Descomenta para entrenar
    # model = train(config="medium", save_path="vae_minecraft_medium.pth")
    
    # ========================================
    # MODO 2: Entrenar TODAS las configuraciones
    # ========================================
    print("\n" + "="*80)
    print("🚀 MODO: Entrenar todas las configuraciones")
    print("="*80)
    models_trained = train_all_configs()
    
    # ========================================
    # MODO 3: Cargar y probar un modelo
    # ========================================
    print("\n" + "="*80)
    print("🔍 MODO: Probar modelo existente")
    print("="*80)
    
    # Cargar modelo
    try:
        model = load_model("vae_minecraft.pth")
    except FileNotFoundError:
        print("⚠️  No se encontró el modelo. Entrenando uno nuevo...")
        import time
        start_time = time.time()
        model = train(config="medium", save_path="vae_minecraft.pth")
        end_time = time.time()
        print(f"⏱️  Tiempo total de entrenamiento: {(end_time - start_time)/60:.2f} minutos")
    
    # Probar con imagen de ejemplo
    prediction_example = "src/metrics/example.png"
    if os.path.exists(prediction_example):
        print(f"\n🖼️  Probando con: {prediction_example}")
        prediction_result = predict(model, prediction_example)
        visualize_prediction(prediction_result)
    
    # Probar con imágenes aleatorias del dataset
    screenshots_dir = "src/metrics/agent_metrics/A3_screenshots"
    image_files = glob.glob(os.path.join(screenshots_dir, "*.png")) + \
                  glob.glob(os.path.join(screenshots_dir, "*.jpg"))
    
    if image_files:
        print(f"\n🎲 Probando con 3 imágenes aleatorias del dataset...")
        for i in range(min(3, len(image_files))):
            random_image = random.choice(image_files)
            print(f"\n{i+1}. {os.path.basename(random_image)}")
            result = predict(model, random_image)
            visualize_prediction(result, title=f"Predicción {i+1}: {os.path.basename(random_image)}")
    else:
        print(f"⚠️  No se encontraron imágenes en {screenshots_dir}")
    
    # ========================================
    # MODO 4: Comparar múltiples modelos
    # ========================================
    print("\n" + "="*80)
    print("📊 MODO: Comparar modelos")
    print("="*80)
    # 
    models_to_compare = {
        "Small": "vae_minecraft_small.pth",
        "Medium": "vae_minecraft_medium.pth",
        "Large": "vae_minecraft_large.pth",
    }
    # 
    # Seleccionar imágenes de prueba
    test_images = random.sample(image_files, min(3, len(image_files)))
    
    # Comparar modelos
    compare_models(models_to_compare, test_images, save_path="vae_comparison.png")
    # 
    # # Evaluar cada modelo
    for name, path in models_to_compare.items():
        if os.path.exists(path):
            model = load_model(path)
            metrics = evaluate_model(model, test_images[:10], model_name=name)
    
    print("\n" + "="*80)
    print("✅ COMPLETADO")
    print("="*80)
    print("\n📚 Para usar el modelo:")
    print("  model = load_model('vae_minecraft.pth')")
    print("  result = predict(model, 'ruta/a/imagen.png')")
    print("  visualize_prediction(result)")
    print("\n🔧 Para entrenar todas las configuraciones:")
    print("  models_trained = train_all_configs()")
    print("\n📊 Para comparar modelos:")
    print("  compare_models({'Model1': 'path1.pth', 'Model2': 'path2.pth'}, test_images)")


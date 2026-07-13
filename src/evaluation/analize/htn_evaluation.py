#!/usr/bin/env python3
"""
Script simple para analizar métricas de agentes HTN
"""

'''
Para los iconos y otros elementos de minecraft mirar: https://mcicons.ccleaf.com/
'''

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # Para imágenes en gráficas 
import numpy as np  # Para arrays
from PIL import Image  # Para redimensionar iconos
import seaborn as sns
from datetime import datetime
import os
from matplotlib.patches import Patch

# Variable global para cachear si los iconos están disponibles
_icons_checked = False
_icons_available = False
_icon_cache = {}  # Cache de iconos ya cargados y optimizados

def load_and_optimize_icon(icon_path, target_size=512):
    """Carga una imagen PNG, la redimensiona si es muy grande, y la convierte a array numpy"""
    try:
        # Verificar si ya está en cache
        if icon_path in _icon_cache:
            return _icon_cache[icon_path]
        
        # Cargar imagen con PIL
        img = Image.open(icon_path)
        
        # Si es muy grande, redimensionar manteniendo aspecto
        if img.width > target_size or img.height > target_size:
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convertir a array numpy que matplotlib entiende
        img_array = np.array(img)
        
        # Guardar en cache
        _icon_cache[icon_path] = img_array
        
        return img_array
    except Exception as e:
        print(f"[ERROR] No se pudo cargar icono {icon_path}: {e}")
        return None

def setup_icons_dir(icons_dir='src/evaluation/icons/'):
    """Configura el directorio de iconos y avisa al usuario si no existen"""
    global _icons_checked, _icons_available
    
    if _icons_checked:
        return _icons_available
    
    _icons_checked = True
    icons_path = Path(icons_dir)
    
    if not icons_path.exists():
        print(f"\n[AVISO] Directorio de iconos no encontrado: {icons_dir}")
        print("Para descargar los iconos de Minecraft:")
        print("  1. Visita: https://mcicons.ccleaf.com/")
        print(f"  2. Descarga los iconos en: {icons_dir}/")
        print("  3. Las gráficas funcionarán sin iconos pero con mejor aspecto con ellos\n")
        return False
    
    # Verificar si hay algún PNG en el directorio
    png_files = list(icons_path.glob('*.png'))
    _icons_available = len(png_files) > 0
    
    if not _icons_available:
        print(f"\n[AVISO] No se encontraron archivos PNG en {icons_dir}")
        print("Las gráficas funcionarán sin iconos\n")
    
    return _icons_available

def get_minecraft_icon(action_name, icons_dir='src/evaluation/icons/'):
    """Mapea nombre de acción a archivo PNG de Minecraft y lo optimiza"""
    mapping = {
        "Madera": "Oak_Log.png",      # Tronco de roble
        "Tablones": "Oak_Planks.png", # Tablones de roble
        "Palos": "Stick.png",
        "Asegurar Mesa": "Crafting_Table.png",  # Crafting table
        "Pico Madera": "Wooden_Pickaxe.png",
        "Piedra Total": "Cobblestone.png",  # Piedra/cobblestone
        "Pico Piedra": "Stone_Pickaxe.png",
        "Carbón": "Coal.png",
        "Hierro": "Raw_Iron.png",
        "Craftear Horno": "Furnace.png",  # Preview horno
        "Colocar Horno": "Furnace.png",
        "Lingote de Hierro": "Iron_Ingot.png",
        "Pico de Hierro": "Iron_Pickaxe.png"
    }
    
    if not _icons_available:
        return None
    
    filename = mapping.get(action_name, "Cancel.png")  # Default
    path = os.path.join(icons_dir, filename)
    
    if os.path.exists(path):
        return load_and_optimize_icon(path, target_size=80)
    return None



# Configurar estilo de gráficas
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_metrics(metrics_dir):
    """Cargar todos los archivos de métricas"""
    metrics = []
    metrics_path = Path(metrics_dir)
    
    for file in metrics_path.glob("metrics_*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            metrics.append(data)
    
    print(f"Cargados {len(metrics)} archivos de métricas")
    return metrics

def analyze_basic_stats(metrics):
    """Análisis básico de estadísticas"""
    print("\n" + "="*60)
    print("ESTADÍSTICAS BÁSICAS")
    print("="*60)
    
    total = len(metrics)
    successful = sum(1 for m in metrics if m.get('success', False))
    failed = total - successful
    
    print(f"\nTotal de ejecuciones: {total}")
    print(f"Exitosas: {successful} ({successful/total*100:.1f}%)")
    print(f"Fallidas: {failed} ({failed/total*100:.1f}%)")
    
    # Tiempos
    times = [m['time_elapsed_s'] for m in metrics]
    print(f"\nTIEMPOS DE EJECUCIÓN:")
    print(f"   Promedio: {sum(times)/len(times):.2f}s")
    print(f"   Mínimo: {min(times):.2f}s")
    print(f"   Máximo: {max(times):.2f}s")
    
    # Distancias
    distances = [m['exploration_distance'] for m in metrics]
    print(f"\nDISTANCIA DE EXPLORACIÓN:")
    print(f"   Promedio: {sum(distances)/len(distances):.2f} bloques")
    print(f"   Mínima: {min(distances):.2f} bloques")
    print(f"   Máxima: {max(distances):.2f} bloques")

def plot_action_timeline(metrics):
    fig, ax = plt.subplots(figsize=(15, 8)) 
    max_time = 0
    
    # Parámetros para el control de solapamiento
    TIME_THRESHOLD = 15  # Segundos mínimos para considerar que hay solapamiento
    OFFSET_VALUE = 0.25  # Cuánto se desplaza el icono verticalmente (0.25 de la unidad del eje Y)

    for idx, m in enumerate(metrics):
        agent = m['agent_name']
        start = datetime.fromisoformat(m['start_time'].replace('Z', '+00:00'))
        
        # Guardamos los tiempos de las acciones ya colocadas para este agente
        placed_times = []
        
        for action in m['actions']:
            action_time = datetime.fromisoformat(action['timestamp'].replace('Z', '+00:00'))
            elapsed = (action_time - start).total_seconds()
            max_time = max(max_time, elapsed)
            
            # --- Lógica Antisolapamiento ---
            # Si hay alguna acción previa muy cerca, alternamos el offset
            current_offset = 0
            # Contamos cuántas acciones hay en el mismo rango de tiempo
            nearby_actions = sum(1 for t in placed_times if abs(t - elapsed) < TIME_THRESHOLD)
            
            if nearby_actions > 0:
                # Alterna: primera cerca arriba (+0.25), segunda cerca abajo (-0.25), tercera (+0.5)...
                direction = 1 if nearby_actions % 2 != 0 else -1
                current_offset = direction * OFFSET_VALUE * ((nearby_actions + 1) // 2)
            
            placed_times.append(elapsed)
            # -------------------------------

            img_array = get_minecraft_icon(action['name'])
            y_pos = idx + current_offset # Aplicamos el desplazamiento vertical
            
            if img_array is not None:
                try:
                    # He bajado el zoom a 0.35 para que respiren mejor
                    img = OffsetImage(img_array, zoom=0.35, alpha=1.0) 
                    ab = AnnotationBbox(img, (elapsed, y_pos), frameon=False, pad=0)
                    ax.add_artist(ab)
                except Exception:
                    ax.scatter(elapsed, y_pos, s=100, c='gray')
            else:
                ax.scatter(elapsed, y_pos, s=100, c='gray')

    ax.set_xlim(-max_time * 0.05, max_time * 1.1) # Un poco más de margen a la derecha
    ax.set_ylim(-1, len(metrics))
    
    ax.set_xlabel('Tiempo (segundos)', fontsize=12)
    ax.set_ylabel('Agentes', fontsize=12)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m['agent_name'] for m in metrics])
    
    # Añadimos líneas horizontales suaves para cada agente como referencia
    for i in range(len(metrics)):
        ax.axhline(i, color='gray', linestyle='-', alpha=0.1, zorder=0)

    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    plt.tight_layout()
    
    return fig

def plot_exploration_distance(metrics):
    """Gráfica de distancia de exploración"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    agents = [m['agent_name'] for m in metrics]
    distances = [m['exploration_distance'] for m in metrics]
    times = [m['time_elapsed_s'] for m in metrics]
    colors = ['green' if m['success'] else 'red' for m in metrics]
    
    scatter = ax.scatter(times, distances, c=colors, s=200, alpha=0.8)
    
    # Etiquetas
    for i, agent in enumerate(agents):
        ax.annotate(agent, (times[i], distances[i]), 
                   fontsize=9, ha='center', va='bottom')
    
    ax.set_xlabel('Tiempo de Ejecución (segundos)')
    ax.set_ylabel('Distancia Explorada (bloques)')
    ax.set_title('Distancia de Exploración vs Tiempo')
    ax.grid(True, alpha=0.3)
    
    # Leyenda
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Exitoso'),
        Patch(facecolor='red', alpha=0.6, label='Fallido')
    ]
    ax.legend(handles=legend_elements)
    
    fig.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.1)
    return fig

def main():
    """Función principal"""
    # Configurar iconos
    setup_icons_dir()
    
    # Ruta a las métricas
    metrics_dir = Path(__file__).parent.parent / "agents" / "metrics"
    
    if not metrics_dir.exists():
        print(f"No se encontró el directorio: {metrics_dir}")
        return
    
    # Cargar métricas
    metrics = load_metrics(metrics_dir)
    
    if not metrics:
        print("No se encontraron archivos de métricas")
        return
    
    # Análisis básico
    analyze_basic_stats(metrics)
    
    # Crear directorio para gráficas
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerando gráficas...")
    
    # Gráfica 2: Timeline de acciones
    fig2 = plot_action_timeline(metrics)
    fig2.savefig(output_dir / "action_timeline.png", dpi=150, bbox_inches='tight')
    print(f"   Guardada: /src/evaluation/plots/action_timeline.png")
    
    # Gráfica 3: Exploración
    fig3 = plot_exploration_distance(metrics)
    fig3.savefig(output_dir / "exploration_distance.png", dpi=150, bbox_inches='tight')
    print(f"   Guardada: /src/evaluation/plots/exploration_distance.png")
    
    print(f"\nAnálisis completado. Gráficas guardadas en: {output_dir}")
    print("\nPara ver las gráficas interactivamente, descomenta 'plt.show()' al final")
    
    # Descomentar para ver gráficas interactivas
    # plt.show()

if __name__ == "__main__":
    main()

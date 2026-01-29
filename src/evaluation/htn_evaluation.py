#!/usr/bin/env python3
"""
Script simple para analizar métricas de agentes HTN
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

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

def plot_execution_times(metrics):
    """Gráfica de tiempos de ejecución"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica 1: Tiempo por agente
    agents = [m['agent_name'] for m in metrics]
    times = [m['time_elapsed_s'] for m in metrics]
    colors = ['green' if m['success'] else 'red' for m in metrics]
    
    ax1.bar(agents, times, color=colors, alpha=0.7)
    ax1.set_xlabel('Agente')
    ax1.set_ylabel('Tiempo (segundos)')
    ax1.set_title('Tiempo de Ejecución por Agente')
    ax1.tick_params(axis='x', rotation=45)
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Exitoso'),
        Patch(facecolor='red', alpha=0.7, label='Fallido')
    ]
    ax1.legend(handles=legend_elements)
    
    # Gráfica 2: Box plot de tiempos exitosos vs fallidos
    success_times = [m['time_elapsed_s'] for m in metrics if m['success']]
    fail_times = [m['time_elapsed_s'] for m in metrics if not m['success']]
    
    data_to_plot = []
    labels = []
    if success_times:
        data_to_plot.append(success_times)
        labels.append(f'Exitoso\n(n={len(success_times)})')
    if fail_times:
        data_to_plot.append(fail_times)
        labels.append(f'Fallido\n(n={len(fail_times)})')
    
    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        ax2.set_ylabel('Tiempo (segundos)')
        ax2.set_title('Distribución de Tiempos')
    
    plt.tight_layout()
    return fig

def plot_action_timeline(metrics):
    """Timeline de acciones"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for idx, m in enumerate(metrics):
        agent = m['agent_name']
        start = datetime.fromisoformat(m['start_time'].replace('Z', '+00:00'))
        
        for action in m['actions']:
            action_time = datetime.fromisoformat(action['timestamp'].replace('Z', '+00:00'))
            elapsed = (action_time - start).total_seconds()
            ax.scatter(elapsed, idx, s=200, alpha=0.8)
            ax.text(elapsed, idx, action['name'], fontsize=8, 
                   rotation=45, ha='left', va='bottom')
    
    ax.set_xlabel('Tiempo desde inicio (segundos)')
    ax.set_ylabel('Agente')
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m['agent_name'] for m in metrics])
    ax.set_title('Timeline de Acciones por Agente')
    ax.grid(True, alpha=0.3)
    
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
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Exitoso'),
        Patch(facecolor='red', alpha=0.6, label='Fallido')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    return fig

def main():
    """Función principal"""
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
    
    # Gráfica 1: Tiempos de ejecución
    fig1 = plot_execution_times(metrics)
    fig1.savefig(output_dir / "execution_times.png", dpi=150, bbox_inches='tight')
    print(f"   Guardada: /src/evaluation/plots/execution_times.png")
    
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

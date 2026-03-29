import json
from collections import Counter
import os

INPUT_FILE = "data/train.jsonl"
OUTPUT_FILE = "data/train_cleaned.jsonl"

# Configuración
MIN_SAMPLES_THRESHOLD = 50  # Eliminaremos clases con menos de 50 ejemplos
FRAME_SKIP = 3              # Nos quedaremos con 1 de cada 3 fotogramas

def clean_dataset():
    print(f"--- Iniciando limpieza de {INPUT_FILE} ---")
    
    # Paso 1: Leer todo y contar frecuencias reales de las acciones
    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()
        
    # Extraer las acciones para contar (asumiendo que action es un string o dict con 'name')
    actions = []
    for line in lines:
        data = json.loads(line)
        action = data.get("action", "idle")
        if isinstance(action, dict):
            action = action.get("name", "idle")
        actions.append(action)
        
    counts = Counter(actions)
    print("\nFrecuencias originales:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
        
    valid_classes = {k for k, v in counts.items() if v >= MIN_SAMPLES_THRESHOLD}
    print(f"\nClases que sobreviven (>= {MIN_SAMPLES_THRESHOLD} muestras): {len(valid_classes)}")

    # Paso 2: Filtrar y aplicar Frame Skip
    cleaned_lines = []
    skipped_by_class = 0
    
    for i, line in enumerate(lines):
        data = json.loads(line)
        action = data.get("action", "idle")
        if isinstance(action, dict):
            action = action.get("name", "idle")
            
        # Filtro 1: ¿Pertenece a una clase válida?
        if action not in valid_classes:
            skipped_by_class += 1
            continue
            
        # Filtro 2: Frame skip (Subsampling temporal)
        # Solo guardamos si el índice es múltiplo de FRAME_SKIP
        if i % FRAME_SKIP == 0:
            cleaned_lines.append(line)

    # Paso 3: Guardar el nuevo dataset
    with open(OUTPUT_FILE, 'w') as f:
        f.writelines(cleaned_lines)
        
    print(f"\n--- Resumen ---")
    print(f"Líneas originales: {len(lines)}")
    print(f"Líneas eliminadas por clase minoritaria: {skipped_by_class}")
    print(f"Líneas eliminadas por frame skip: {len(lines) - skipped_by_class - len(cleaned_lines)}")
    print(f"Líneas finales guardadas en {OUTPUT_FILE}: {len(cleaned_lines)}")

if __name__ == "__main__":
    clean_dataset()
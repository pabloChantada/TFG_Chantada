import json
import os
import glob
from pathlib import Path
from datetime import datetime

def check_screenshots_in_metrics(metrics_file):
    """
    Verifica si todas las acciones en un archivo metrics.json tienen screenshot.
    
    Args:
        metrics_file: Ruta al archivo metrics.json
        
    Returns:
        dict con información sobre las acciones sin screenshot
    """
    with open(metrics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    agent_name = data.get('agent_name', 'Unknown')
    
    # Buscar acciones en control_tracking.rl_actions.action_sequence
    rl_actions = data.get('control_tracking', {}).get('rl_actions', {})
    action_sequence = rl_actions.get('action_sequence', [])
    
    total_actions = len(action_sequence)
    actions_with_screenshot = 0
    actions_without_screenshot = []
    missing_screenshots = []
    timestamp_gaps = []
    
    # Screenshot paths in the JSON are relative to the project root (e.g.
    # "src/metrics/agent_metrics/<agent>_screenshots/..."), NOT relative to
    # the metrics file itself, so we resolve from CWD.
    base_dir = Path.cwd()
    
    prev_timestamp = None
    for i, action in enumerate(action_sequence):
        screenshot_path = action.get('screenshot')
        current_timestamp = action.get('timestamp')
        
        # Calcular diferencia entre timestamps
        if prev_timestamp and current_timestamp:
            try:
                prev_dt = datetime.fromisoformat(prev_timestamp.replace('Z', '+00:00'))
                curr_dt = datetime.fromisoformat(current_timestamp.replace('Z', '+00:00'))
                time_diff = (curr_dt - prev_dt).total_seconds()
                
                timestamp_gaps.append({
                    'index': i,
                    'from_timestamp': prev_timestamp,
                    'to_timestamp': current_timestamp,
                    'gap_seconds': time_diff
                })
            except Exception as e:
                pass  # Ignorar errores de parsing
        
        prev_timestamp = current_timestamp
        screenshot_path = action.get('screenshot')
        
        if screenshot_path:
            actions_with_screenshot += 1
            # Verificar si el archivo existe
            full_path = base_dir / screenshot_path
            if not full_path.exists():
                missing_screenshots.append({
                    'index': i,
                    'timestamp': action.get('timestamp'),
                    'path': screenshot_path
                })
        else:
            actions_without_screenshot.append({
                'index': i,
                'timestamp': action.get('timestamp'),
                'position': action.get('position')
            })
    
    # Analizar gaps en timestamps
    avg_gap = sum(g['gap_seconds'] for g in timestamp_gaps) / len(timestamp_gaps) if timestamp_gaps else 0
    max_gap = max((g['gap_seconds'] for g in timestamp_gaps), default=0)
    min_gap = min((g['gap_seconds'] for g in timestamp_gaps), default=0)
    
    # Detectar gaps sospechosos (más de 2x el promedio o más de 5 segundos)
    threshold = max(2 * avg_gap, 5.0) if avg_gap > 0 else 5.0
    large_gaps = [g for g in timestamp_gaps if g['gap_seconds'] > threshold]
    
    return {
        'agent_name': agent_name,
        'metrics_file': metrics_file,
        'total_actions': total_actions,
        'actions_with_screenshot': actions_with_screenshot,
        'actions_without_screenshot': len(actions_without_screenshot),
        'missing_screenshot_files': len(missing_screenshots),
        'actions_without_screenshot_details': actions_without_screenshot,
        'missing_screenshot_files_details': missing_screenshots,
        'percentage_with_screenshot': (actions_with_screenshot / total_actions * 100) if total_actions > 0 else 0,
        'timestamp_analysis': {
            'total_gaps': len(timestamp_gaps),
            'avg_gap_seconds': avg_gap,
            'min_gap_seconds': min_gap,
            'max_gap_seconds': max_gap,
            'large_gaps_count': len(large_gaps),
            'large_gaps_details': large_gaps,
            'threshold_seconds': threshold
        }
    }

def check_all_metrics():
    """
    Verifica todos los archivos *_metrics.json en la carpeta agent_metrics.
    """
    metrics_dir = Path(__file__).parent.parent / 'metrics' / 'agent_metrics'
    metrics_files = sorted(glob.glob(str(metrics_dir / '*_metrics.json')))
    
    if not metrics_files:
        print(f"❌ No se encontraron archivos metrics.json en {metrics_dir}")
        return
    
    print(f"📊 Analizando {len(metrics_files)} archivo(s) de métricas...\n")
    print("=" * 80)
    
    all_results = []
    
    for metrics_file in metrics_files:
        result = check_screenshots_in_metrics(metrics_file)
        all_results.append(result)
        
        print(f"\n🤖 Agente: {result['agent_name']}")
        print(f"📄 Archivo: {Path(metrics_file).name}")
        print(f"📸 Total de acciones: {result['total_actions']}")
        print(f"✅ Con screenshot: {result['actions_with_screenshot']} ({result['percentage_with_screenshot']:.2f}%)")
        print(f"❌ Sin screenshot: {result['actions_without_screenshot']}")
        print(f"🔍 Screenshots faltantes (archivo no existe): {result['missing_screenshot_files']}")
        
        # Mostrar análisis de timestamps
        ts_analysis = result['timestamp_analysis']
        print(f"\n⏱️  Análisis de Timestamps:")
        print(f"   - Total de gaps: {ts_analysis['total_gaps']}")
        print(f"   - Gap promedio: {ts_analysis['avg_gap_seconds']:.3f}s")
        print(f"   - Gap mínimo: {ts_analysis['min_gap_seconds']:.3f}s")
        print(f"   - Gap máximo: {ts_analysis['max_gap_seconds']:.3f}s")
        print(f"   - Gaps grandes (>{ts_analysis['threshold_seconds']:.1f}s): {ts_analysis['large_gaps_count']}")
        
        # Mostrar detalles de gaps grandes
        if ts_analysis['large_gaps_count'] > 0:
            print(f"\n⚠️  Gaps grandes detectados (posibles acciones saltadas):")
            for gap in ts_analysis['large_gaps_details'][:5]:
                print(f"   - Índice {gap['index']}: {gap['gap_seconds']:.3f}s de diferencia")
            if ts_analysis['large_gaps_count'] > 5:
                print(f"   ... y {ts_analysis['large_gaps_count'] - 5} más")
        
        # Mostrar detalles de acciones sin screenshot
        if result['actions_without_screenshot'] > 0:
            print(f"\n⚠️  Acciones sin screenshot:")
            for action in result['actions_without_screenshot_details'][:5]:  # Mostrar solo las primeras 5
                print(f"   - Índice {action['index']}: {action['timestamp']}")
            if result['actions_without_screenshot'] > 5:
                print(f"   ... y {result['actions_without_screenshot'] - 5} más")
        
        # Mostrar detalles de screenshots faltantes
        if result['missing_screenshot_files'] > 0:
            print(f"\n⚠️  Screenshots con ruta pero archivo no existe:")
            for screenshot in result['missing_screenshot_files_details'][:5]:
                print(f"   - Índice {screenshot['index']}: {screenshot['path']}")
            if result['missing_screenshot_files'] > 5:
                print(f"   ... y {result['missing_screenshot_files'] - 5} más")
        
        print("-" * 80)
    
    # Resumen general
    print(f"\n📋 RESUMEN GENERAL")
    print("=" * 80)
    total_actions = sum(r['total_actions'] for r in all_results)
    total_with_screenshot = sum(r['actions_with_screenshot'] for r in all_results)
    total_without_screenshot = sum(r['actions_without_screenshot'] for r in all_results)
    total_missing_files = sum(r['missing_screenshot_files'] for r in all_results)
    
    print(f"🔢 Total de acciones en todos los archivos: {total_actions}")
    print(f"✅ Acciones con screenshot: {total_with_screenshot} ({total_with_screenshot/total_actions*100:.2f}%)")
    print(f"❌ Acciones sin screenshot: {total_without_screenshot} ({total_without_screenshot/total_actions*100:.2f}%)")
    print(f"🔍 Screenshots faltantes (archivo no existe): {total_missing_files}")
    
    # Análisis de timestamps agregado
    all_gaps = []
    all_large_gaps = []
    for r in all_results:
        ts = r['timestamp_analysis']
        all_gaps.extend([ts['avg_gap_seconds']] * ts['total_gaps'] if ts['total_gaps'] > 0 else [])
        all_large_gaps.extend(ts['large_gaps_details'])
    
    if all_gaps:
        overall_avg_gap = sum(all_gaps) / len(all_gaps)
        print(f"\n⏱️  Gap promedio global: {overall_avg_gap:.3f}s")
        print(f"🚨 Total de gaps grandes detectados: {len(all_large_gaps)}")
        if len(all_large_gaps) > 0:
            print(f"   (Posibles acciones saltadas entre capturas)")
    
    # Verificar si todas tienen screenshot
    if total_without_screenshot == 0 and total_missing_files == 0:
        print("\n🎉 ¡TODAS LAS ACCIONES TIENEN SCREENSHOT Y EXISTEN!")
    elif total_without_screenshot == 0:
        print(f"\n⚠️  Todas las acciones tienen ruta de screenshot, pero faltan {total_missing_files} archivos")
    else:
        print(f"\n⚠️  Hay {total_without_screenshot} acciones sin screenshot")
    
    return all_results

if __name__ == '__main__':
    check_all_metrics()

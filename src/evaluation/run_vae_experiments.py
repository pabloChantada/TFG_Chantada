#!/usr/bin/env python3
"""
Script para ejecutar experimentos con diferentes configuraciones de VAE
Permite entrenar, evaluar y comparar múltiples modelos de forma automatizada
"""

import os
import sys
import argparse
import json
from vae_train import (
    train, load_model, predict, visualize_prediction,
    compare_models, evaluate_model, train_all_configs,
    VAE_CONFIGS, AGENT_METRICS_PATH
)
import glob
import random


def main():
    parser = argparse.ArgumentParser(description="Entrenar y evaluar VAE para screenshots de Minecraft")
    parser.add_argument("--mode", type=str, default="test",
                       choices=["train", "train-all", "test", "compare", "evaluate"],
                       help="Modo de operación")
    parser.add_argument("--config", type=str, default="medium",
                       choices=list(VAE_CONFIGS.keys()),
                       help="Configuración de VAE a usar")
    parser.add_argument("--model", type=str, default="vae_minecraft.pth",
                       help="Path al modelo")
    parser.add_argument("--data", type=str, default=AGENT_METRICS_PATH,
                       help="Path a los datos de entrenamiento")
    parser.add_argument("--test-image", type=str, default=None,
                       help="Path a imagen específica para probar")
    parser.add_argument("--num-samples", type=int, default=3,
                       help="Número de muestras aleatorias para probar")
    
    args = parser.parse_args()
    
    # ========================================
    # MODO: ENTRENAR UN MODELO
    # ========================================
    if args.mode == "train":
        print(f"\n{'='*80}")
        print(f" Entrenando VAE con configuración: {args.config}")
        print(f"{'='*80}\n")
        
        import time
        start_time = time.time()
        model = train(
            screenshot_path=args.data,
            config=args.config,
            save_path=args.model
        )
        end_time = time.time()
        
        print(f"\n Entrenamiento completado!")
        print(f" Tiempo: {(end_time - start_time)/60:.2f} minutos")
        print(f" Modelo guardado en: {args.model}")
    
    # ========================================
    # MODO: ENTRENAR TODAS LAS CONFIGURACIONES
    # ========================================
    elif args.mode == "train-all":
        print(f"\n{'='*80}")
        print(f" Entrenando TODAS las configuraciones de VAE")
        print(f"{'='*80}\n")
        
        models_trained = train_all_configs(base_path=args.data)
        
        print(f"\n Todos los modelos entrenados:")
        for name, path in models_trained.items():
            print(f"   - {name}: {path}")
        
        # Guardar resumen
        with open("vae_models_trained.json", "w") as f:
            json.dump(models_trained, f, indent=2)
        print(f"\n Resumen guardado en: vae_models_trained.json")
    
    # ========================================
    # MODO: PROBAR UN MODELO
    # ========================================
    elif args.mode == "test":
        print(f"\n{'='*80}")
        print(f" Probando modelo: {args.model}")
        print(f"{'='*80}\n")
        
        # Cargar modelo
        if not os.path.exists(args.model):
            print(f"  Modelo no encontrado: {args.model}")
            print(f"  Entrenando nuevo modelo con configuración '{args.config}'...")
            model = train(
                screenshot_path=args.data,
                config=args.config,
                save_path=args.model
            )
        else:
            model = load_model(args.model)
        
        # Probar con imagen específica
        if args.test_image and os.path.exists(args.test_image):
            print(f"\n Probando con: {args.test_image}")
            result = predict(model, args.test_image)
            visualize_prediction(result)
        
        # Probar con muestras aleatorias
        else:
            screenshots_dirs = glob.glob(os.path.join(args.data, "*_screenshots"))
            all_images = []
            for dir_path in screenshots_dirs:
                all_images.extend(glob.glob(os.path.join(dir_path, "*.png")))
                all_images.extend(glob.glob(os.path.join(dir_path, "*.jpg")))
            
            if all_images:
                num_samples = min(args.num_samples, len(all_images))
                print(f"\n Probando con {num_samples} imágenes aleatorias...")
                test_images = random.sample(all_images, num_samples)
                
                for i, img_path in enumerate(test_images):
                    print(f"\n{i+1}. {os.path.basename(img_path)}")
                    result = predict(model, img_path)
                    visualize_prediction(result, 
                                       title=f"Predicción {i+1}: {os.path.basename(img_path)}")
            else:
                print(f"  No se encontraron imágenes en {args.data}")
    
    # ========================================
    # MODO: COMPARAR MODELOS
    # ========================================
    elif args.mode == "compare":
        print(f"\n{'='*80}")
        print(f" Comparando modelos VAE")
        print(f"{'='*80}\n")
        
        # Buscar todos los modelos entrenados
        model_files = glob.glob("vae_minecraft_*.pth")
        if not model_files:
            print("  No se encontraron modelos entrenados.")
            print("  Primero entrena algunos modelos con: --mode train-all")
            return
        
        # Crear diccionario de modelos
        models_dict = {}
        for model_path in model_files:
            name = os.path.basename(model_path).replace("vae_minecraft_", "").replace(".pth", "")
            models_dict[name] = model_path
        
        print(f"Modelos encontrados: {list(models_dict.keys())}")
        
        # Obtener imágenes de prueba
        screenshots_dirs = glob.glob(os.path.join(args.data, "*_screenshots"))
        all_images = []
        for dir_path in screenshots_dirs:
            all_images.extend(glob.glob(os.path.join(dir_path, "*.png")))
        
        if all_images:
            test_images = random.sample(all_images, min(3, len(all_images)))
            compare_models(models_dict, test_images, save_path="vae_comparison.png")
        else:
            print(f"  No se encontraron imágenes para comparar")
    
    # ========================================
    # MODO: EVALUAR MODELOS
    # ========================================
    elif args.mode == "evaluate":
        print(f"\n{'='*80}")
        print(f" Evaluando modelos VAE")
        print(f"{'='*80}\n")
        
        # Buscar todos los modelos entrenados
        model_files = glob.glob("vae_minecraft_*.pth") + ["vae_minecraft.pth"]
        model_files = [f for f in model_files if os.path.exists(f)]
        
        if not model_files:
            print("  No se encontraron modelos entrenados.")
            return
        
        # Obtener imágenes de prueba
        screenshots_dirs = glob.glob(os.path.join(args.data, "*_screenshots"))
        all_images = []
        for dir_path in screenshots_dirs:
            all_images.extend(glob.glob(os.path.join(dir_path, "*.png")))
        
        if not all_images:
            print(f"  No se encontraron imágenes en {args.data}")
            return
        
        test_images = random.sample(all_images, min(20, len(all_images)))
        
        # Evaluar cada modelo
        results = []
        for model_path in model_files:
            name = os.path.basename(model_path).replace("vae_minecraft_", "").replace(".pth", "")
            print(f"\n{'='*60}")
            print(f"Evaluando: {name}")
            print(f"{'='*60}")
            
            model = load_model(model_path)
            metrics = evaluate_model(model, test_images, model_name=name)
            results.append(metrics)
        
        # Guardar resultados
        with open("vae_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f" RESUMEN DE EVALUACIÓN")
        print(f"{'='*80}\n")
        
        # Ordenar por MSE
        results.sort(key=lambda x: x['avg_mse'])
        
        print("Ranking por MSE (menor es mejor):")
        for i, r in enumerate(results):
            print(f"{i+1}. {r['model_name']:15s} - MSE: {r['avg_mse']:.6f} - MAE: {r['avg_mae']:.6f}")
        
        print(f"\n Resultados guardados en: vae_evaluation_results.json")


if __name__ == "__main__":
    main()

# live_agent.py - Test en vivo en Minecraft
import torch
import pyautogui
import mss
import numpy as np
from PIL import Image
import time
import cv2
from main import device, ACTIONS
from load_dataset import MinecraftDataset
from model import MinecraftILModel

# Configuración Minecraft
SCREEN_REGION = (0, 30, 1920, 1040)  # x, y, width, height
FPS = 10  # 100ms/frame
MODEL_PATH = "src/il/runs/20260328_185613/model_r18_20260328_185613.pth"

def load_agent(dataset_path="data/train.jsonl"):
    """Carga modelo entrenado"""
    dataset = MinecraftDataset(dataset_path)
    model = MinecraftILModel(num_actions=len(dataset.pair2id)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, dataset

def capture_screen():
    """Captura región de Minecraft"""
    with mss.mss() as sct:
        monitor = {"top": SCREEN_REGION[1], "left": SCREEN_REGION[0],
                  "width": SCREEN_REGION[2], "height": SCREEN_REGION[3]}
        screenshot = sct.grab(monitor)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

def execute_action(action_label):
    """Ejecuta acción en Minecraft"""
    if action_label == "idle":
        pyautogui.keyUp(['w', 's', 'a', 'd', 'ctrl', 'space', 'shift'])
        return
    
    if "sprint" in action_label:
        pyautogui.keyDown('w')
        pyautogui.keyDown('ctrl')
    elif "move_forward" in action_label:
        pyautogui.keyDown('w')
    elif "move_backward" in action_label:
        pyautogui.keyDown('s')
    elif "move_left" in action_label:
        pyautogui.keyDown('a')
    elif "move_right" in action_label:
        pyautogui.keyDown('d')
    elif "jump" in action_label:
        pyautogui.press('space')
    elif "sneak" in action_label:
        pyautogui.keyDown('shift')
    elif "attack" in action_label:
        pyautogui.click()
    elif "camera_yaw_p" in action_label:
        pyautogui.moveRel(30, 0)  # Girar derecha
    elif "camera_yaw_m" in action_label:
        pyautogui.moveRel(-30, 0)  # Girar izquierda

def main():
    model, dataset = load_agent()
    print("Agente cargado. Abre Minecraft y presiona ENTER")
    input()
    
    frame_time = 1.0 / FPS
    while True:
        start = time.time()
        
        # 1. Captura pantalla
        screen = capture_screen()
        
        # 2. Predicción
        with torch.no_grad():
            img_tensor = dataset.transforms(screen).unsqueeze(0).to(device)
            logits = model(img_tensor)
            pred_id = logits.argmax(1).item()
        
        # 3. Acción
        id2action = {v: k for k, v in dataset.pair2id.items()}
        action = id2action[pred_id]
        print(f"    Acción: {action}")
        
        # 4. Ejecuta
        execute_action(action)
        
        # 5. Libera teclas + espera FPS
        time.sleep(frame_time - (time.time() - start))
        pyautogui.keyUp(['w', 's', 'a', 'd', 'ctrl', 'space', 'shift'])
        
        # ESC para salir
        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == "__main__":
    main()
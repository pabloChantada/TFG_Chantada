from cnn import YOLOTreeDetector
import mss
import numpy as np
from PIL import Image
import gradio as gr
import time

detector = YOLOTreeDetector()

def continuous_detect():
    """Auto-captura cada 1 segundo"""
    monitor = {"top": 100, "left": 100, "width": 1280, "height": 720}
    
    while True:
        with mss.mss() as sct:
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        trees = detector.predict(np.array(img))
        
        info = f"{len(trees)} arboles detectados"
        if trees:
            nearest = min(trees, key=lambda t: t["bbox"][0])
            info += f"\nMas cercano: x:{nearest['bbox'][0]:.0f} y:{nearest['bbox'][1]:.0f} (conf: {nearest['confidence']:.1%})"
        
        annotated = detector.model.predict(np.array(img), save=False, verbose=False)[0].plot()
        
        yield info, Image.fromarray(annotated)
        # time.sleep(0.2)  # 1 FPS continuo, ajustar como veamos

with gr.Blocks(title="Detector Arboles Minecraft CONTINUO") as demo:
    gr.Markdown("# YOLOv8 Detector Arboles MINECRAFT CONTINUO")
    gr.Markdown("Auto-actualiza cada 1s. Minecraft ventana 1280x720 fija.")
    
    with gr.Row():
        info = gr.Textbox(label="Info", lines=4)
        image = gr.Image(label="Screen + Boxes")
    
    start_btn = gr.Button("INICIAR CONTINUO", variant="primary")
    stop_btn = gr.Button("PARAR")
    
    # CONTINUO con yield
    start_btn.click(continuous_detect, outputs=[info, image])
    
    gr.Markdown("Ajusta monitor= linea 12 a tu ventana Minecraft")

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)
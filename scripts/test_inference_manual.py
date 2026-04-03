#!/usr/bin/env python3
"""
test_inference.py — Test online del inference server.

Captura la pantalla de Minecraft en tiempo real y muestra predicciones.

Uso:
  python scripts/test_inference.py                   # captura pantalla completa
  python scripts/test_inference.py --monitor 2       # monitor secundario
  python scripts/test_inference.py --interval 0.5    # cada 0.5s
"""

import sys
import json
import time
import argparse
import http.client
from io import BytesIO

import mss
from PIL import Image


def send_predict(png_bytes, state_str="0,0,0,0,0,0,0", host="localhost", port=8765):
    conn = http.client.HTTPConnection(host, port, timeout=10)
    headers = {
        "Content-Type": "image/png",
        "Content-Length": str(len(png_bytes)),
        "X-Bot-State": state_str,
    }
    conn.request("POST", "/predict", body=png_bytes, headers=headers)
    resp = conn.getresponse()
    data = json.loads(resp.read().decode())
    conn.close()
    return data


def capture_to_png(sct, monitor):
    raw = sct.grab(monitor)
    img = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def print_result(result, step, latency_ms):
    action = result["action"]
    conf   = result["confidence"]
    cam    = result.get("camera_delta", {})
    dyaw   = cam.get("dyaw", 0)
    dpitch = cam.get("dpitch", 0)

    # Limpiar lineas anteriores (8 lineas: header + action + camera + 5 top5)
    if step > 1:
        print(f"\033[8A\033[J", end="")

    print(f"  step {step}  ({latency_ms:.0f}ms)")
    print(f"  Action : {action}  ({conf:.1%})")
    print(f"  Camera : dyaw={dyaw:+.4f}  dpitch={dpitch:+.4f}")
    for name, prob in result.get("top5", []):
        bar = "#" * int(prob * 30)
        print(f"    {name:<25} {prob:.1%}  {bar}")


def main():
    p = argparse.ArgumentParser(description="Test online del inference server")
    p.add_argument("--monitor", type=int, default=1,
                   help="Indice del monitor (1=principal, 2=secundario, ...)")
    p.add_argument("--interval", type=float, default=0.8,
                   help="Intervalo entre capturas en segundos (default: 0.8)")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    print(f"Capturando monitor {args.monitor} cada {args.interval}s")
    print(f"Servidor: localhost:{args.port}/predict")
    print(f"Ctrl+C para salir\n")

    step = 0
    with mss.mss() as sct:
        if args.monitor >= len(sct.monitors):
            print(f"Error: monitor {args.monitor} no existe (hay {len(sct.monitors)-1})")
            sys.exit(1)
        monitor = sct.monitors[args.monitor]

        while True:
            try:
                t0 = time.time()
                png_bytes = capture_to_png(sct, monitor)
                result = send_predict(png_bytes, port=args.port)
                latency_ms = (time.time() - t0) * 1000
                step += 1
                print_result(result, step, latency_ms)

                elapsed = time.time() - t0
                wait = args.interval - elapsed
                if wait > 0:
                    time.sleep(wait)

            except ConnectionRefusedError:
                print(f"\nError: no se pudo conectar a localhost:{args.port}")
                print("Asegurate de que el inference server esta corriendo.")
                sys.exit(1)
            except KeyboardInterrupt:
                print(f"\n\nDetenido tras {step} steps.")
                break


if __name__ == "__main__":
    main()

## Flujo 

1. Abrir el mundo de Minecraft en LAN en el puerto deseado.

2. Generar dataset con el agente grabador:

   `node src/evaluation/mass_record.js --episodes 20 --minecraft-port 55916`

   Si quieres balancear automáticamente:

   `node src/evaluation/mass_record.js --episodes 20 --minecraft-port 55916 --balance`

3. Entrenar el modelo de imitation learning (behavioral cloning):

   Con dataset normal:

   `python src/rl/train.py --mode bc --jsonl data/train.jsonl --model_path models/bc_minecraft.pt`

   Con dataset balanceado:

   `python src/rl/train.py --mode bc --jsonl data/train_balanced.jsonl --model_path models/bc_minecraft.pt`

4. Lanzar backend JS del entorno para evaluación:

   `node src/rl/server.js --mc_port 55916`

5. Evaluar el modelo entrenado:

   `python src/rl/train.py --mode bc_eval --model_path models/bc_minecraft.pt --server http://localhost:3001 --episodes 10`

### Variante con espacio de acciones simple

Añadir `--simple` tanto en entrenamiento como en evaluación:

- `python src/rl/train.py --mode bc --jsonl data/train.jsonl --model_path models/bc_minecraft.pt --simple`
- `python src/rl/train.py --mode bc_eval --model_path models/bc_minecraft.pt --server http://localhost:3001 --episodes 10 --simple`

### Archivos clave en este flujo

- `train.py`: entrenamiento y evaluación de behavioral cloning (CNN + estado).
- `env.py`: wrapper Gym (`MinecraftWoodEnv`, `MinecraftWoodSimpleEnv`).
- `server.js`: API HTTP (`/reset`, `/step`, `/state`, `/info`, `/close`).
- `actions.js`: aplicación de acciones al bot.
- `state.js`: extracción de observación.
- `mass_record.js`: grabación masiva de episodios para construir el dataset.

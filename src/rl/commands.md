## Flujo 

1. Lanzar backend JS del entorno:

   `node src/rl/server.js --mc_port 55916`

2. Entrenar/evaluar desde Python con Gym:

   `python src/rl/train.py --mode ppo --server http://localhost:3001`

### Archivos clave en este flujo

- `train.py`: entrenamiento/evaluación con Gymnasium + SB3.
- `env.py`: wrapper Gym (`MinecraftWoodEnv`, `MinecraftWoodSimpleEnv`).
- `server.js`: API HTTP (`/reset`, `/step`, `/state`, `/info`, `/close`).
- `actions.js`: aplicación de acciones al bot.
- `state.js`: extracción de observación.
- `reward.js`: función de recompensa.

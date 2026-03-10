# 1. Servidor JS (en un terminal)
node src/rl/server.js --mc_port 55916

# 2. Entrenamiento Python (en otro terminal)
python src/rl/train.py --mode random --episodes 10 --simple
python src/rl/train.py --mode ppo --timesteps 50000 --simple
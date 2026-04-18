python src/rl/train.py --episodes 200 --eps-decay 30000 --target-update 500 --reset-world-every 20 --max-steps 300

node src/agents/add_agent.js --type rl --name RLBot --server-dir server --use-seed
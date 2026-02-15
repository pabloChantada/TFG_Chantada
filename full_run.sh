#!/bin/bash
# Levantar el server en background
./test_server.sh &
SERVER_PID=$!

echo "Server iniciado con PID $SERVER_PID"

# Esperar a que el server esté listo
sleep 3

# Ejecutar los agentes
./test_agents.sh --type htn --names A8 --port 8080

# Cuando terminan los agentes, matar el server
kill $SERVER_PID

echo "Server detenido"

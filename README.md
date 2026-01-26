# Simbolic Minecraft

## Estructura
- `src/htn/`: lógica HTN + primitivas (recolección, crafteo, colocación, fundición, etc.).
- `src/llm/`: fork de Mindcraft (servidor + viewer + perfiles), usado para conectar bots.
- `api_test/`: scripts sueltos de pruebas.

## Requisitos
- Node.js v18 o v20 LTS.
- Un mundo/servidor de Minecraft Java local. Usando un mundo abierto a LAN.
- Build tools. En Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3 make g++ \
	libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev
```

## Instalación

Dos zonas con dependencias Node:

1) Dependencias del repo raíz (HTN y utilidades)

```bash
cd /TFG_Chantada
npm install
```

2) Dependencias de Mindcraft

```bash
cd /TFG_Chantada/src/llm
npm install
```

> Nota: `src/llm/` usa `patch-package` en `postinstall`, así que el `npm install` aplica parches automáticamente.

## Ejecución (Mindcraft + HTN)

### 1) Arrancar el mundo de Minecraft

- Crear un mundo en local y abrir a LAN.
- El host/puerto configurado en Mindcraft coincide con el mundo. (Por defecto está seteado a 25565)

### 2) Lanzar el servidor

```bash
cd /TFG_Chantada/src/llm
node main.js
```

Esto levanta:

- Viewer de prismarine en `http://localhost:3000`
- Inventario web en `http://localhost:3001`
- MindServer en `http://localhost:8080`
- Lanza un agente automaticamente

### 3) Lanza agentes extra

```bash
cd /TFG_Chantada/src/llm/my_agent
node start.js -n HTNAgent -p 8080 -m "start htn" -c 0
```

El HTN se ejecuta desde `src/htn/main_htn.js` y corre la progresión hacia pico de hierro.

## Notas de funcionamiento

- El crafteo que requiere mesa se gestiona moviéndose a rango de interacción (≈3 bloques).
- La recolección y minería usan búsqueda por `findBlock` con exploración (enfoque “omnisciente”), evitando heurísticas complejas.

## Referencias

- Mineflayer: https://github.com/PrismarineJS/mineflayer
- mineflayer-pathfinder: https://github.com/PrismarineJS/mineflayer-pathfinder
- mineflayer-collectblock: https://github.com/TheDudeFromCI/mineflayer-collectblock
- prismarine-viewer: https://github.com/PrismarineJS/prismarine-viewer
- mineflayer-web-inventory: https://github.com/ImHarvol/mineflayer-web-inventory
- Mindcraft (upstream): https://github.com/mindcraft-bots/mindcraft

## License

MIT License - see [LICENSE](LICENSE)
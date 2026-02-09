#!/usr/bin/env bash
set -euo pipefail

TYPE="htn"
PORT="8080"
MC_PORT="25565"
NAME=""
NAMES=""
COUNT=""
LOAD_MEMORY="false"
INIT_MESSAGE=""
METRICS_PATH="src/metrics/agent_metrics"

print_usage() {
	cat <<'EOF'
Usage: ./test_agents.sh [options]

Options:
  --type htn|llm           Agent type (default: htn)
  --name NAME              Single agent name
  --names N1,N2            Comma-separated agent names (multiple agents)
  --count N                Agent index (single agent only)
  --port N                 MindServer port (default: 8080)
  --minecraft-port N       Minecraft server port (default: 25565)
  --load-memory            Load previous memory
  --init-message MSG       Initial message
  --metrics-path PATH      Metrics export path
  --help                   Show this help

Examples:
  ./test_agents.sh --name Agent1
  ./test_agents.sh --type htn --names A1,A2 --port 8080
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--type)
			TYPE="$2"
			shift 2
			;;
		--name)
			NAME="$2"
			shift 2
			;;
		--names)
			NAMES="$2"
			shift 2
			;;
		--count)
			COUNT="$2"
			shift 2
			;;
		--port)
			PORT="$2"
			shift 2
			;;
		--minecraft-port)
			MC_PORT="$2"
			shift 2
			;;
		--load-memory)
			LOAD_MEMORY="true"
			shift 1
			;;
		--init-message)
			INIT_MESSAGE="$2"
			shift 2
			;;
		--metrics-path)
			METRICS_PATH="$2"
			shift 2
			;;
		--help)
			print_usage
			exit 0
			;;
		*)
			echo "Unknown option: $1" >&2
			print_usage
			exit 1
			;;
	esac
done

if [[ -n "$NAMES" ]]; then
	IFS=',' read -r -a NAME_LIST <<< "$NAMES"
	for i in "${!NAME_LIST[@]}"; do
		AGENT_NAME="${NAME_LIST[$i]}"
		node src/agents/add_agent.js \
			--name "$AGENT_NAME" \
			--type "$TYPE" \
			--port "$PORT" \
			--minecraft-port "$MC_PORT" \
			--count "$i" \
			--metrics-path "$METRICS_PATH" \
			$( [[ "$LOAD_MEMORY" == "true" ]] && echo "--load-memory" ) \
			$( [[ -n "$INIT_MESSAGE" ]] && echo "--init-message" "$INIT_MESSAGE" )
	done
	exit 0
fi

if [[ -z "$NAME" ]]; then
	NAME="Agent1"
fi

if [[ -z "$COUNT" ]]; then
	COUNT="0"
fi

node src/agents/add_agent.js \
	--name "$NAME" \
	--type "$TYPE" \
	--port "$PORT" \
	--minecraft-port "$MC_PORT" \
	--count "$COUNT" \
	--metrics-path "$METRICS_PATH" \
	$( [[ "$LOAD_MEMORY" == "true" ]] && echo "--load-memory" ) \
	$( [[ -n "$INIT_MESSAGE" ]] && echo "--init-message" "$INIT_MESSAGE" )

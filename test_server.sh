#!/usr/bin/env bash
set -euo pipefail

PORT="8080"
NO_UI="true"
CLEAN_METRICS="false"

print_usage() {
    cat <<'EOF'
Usage: ./test_server.sh [options]

Options:
  --port N             Server port (default: 8080)
  --ui                 Enable UI auto-open (default: disabled)
  --clean-metrics      Clear metrics on startup (default: enabled)
  --help               Show this help

Examples:
  ./test_server.sh
  ./test_server.sh --port 9000 --ui
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --ui)
            NO_UI="false"
            shift 1
            ;;
        --clean-metrics)
            CLEAN_METRICS="true"
            shift 1
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

ARGS=("--port" "$PORT")
if [[ "$NO_UI" == "true" ]]; then
    ARGS+=("--no-ui")
fi
if [[ "$CLEAN_METRICS" == "true" ]]; then
    ARGS+=("--clean-metrics")
fi

node src/server/server.js "${ARGS[@]}"

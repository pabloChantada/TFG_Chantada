#!/bin/bash

# Test script for quickly launching and testing agents
# Requires a Minecraft server running on localhost:25565

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
MC_HOST="127.0.0.1"
MC_PORT=25565
MINDSERVER_PORT=8080
CLEANUP_METRICS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            MC_HOST="$2"
            shift 2
            ;;
        -p|--minecraft-port)
            MC_PORT="$2"
            shift 2
            ;;
        --mindserver-port)
            MINDSERVER_PORT="$2"
            shift 2
            ;;
        --cleanup)
            CLEANUP_METRICS=true
            shift
            ;;
        --help)
            echo "Usage: ./test_agents.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -h, --host MC_HOST              Minecraft server host (default: 127.0.0.1)"
            echo "  -p, --minecraft-port PORT       Minecraft server port (default: 25565)"
            echo "  --mindserver-port PORT          MindServer port (default: 8080)"
            echo "  --cleanup                       Clean metrics directory before running"
            echo "  --help                          Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./test_agents.sh                          # Use defaults"
            echo "  ./test_agents.sh --cleanup                # Clean metrics and run"
            echo "  ./test_agents.sh -p 25566 --cleanup       # Custom MC port + cleanup"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to print colored output
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to cleanup metrics
cleanup_metrics() {
    log_info "Cleaning metrics directory..."
    if [ -d "src/metrics/agent_metrics" ]; then
        # Keep example files, remove others
        find src/metrics/agent_metrics -type f ! -name "example_*" -delete
        log_success "Metrics cleaned"
    fi
}

# Function to check if Minecraft server is running
check_minecraft() {
    if timeout 2 bash -c "echo > /dev/tcp/$MC_HOST/$MC_PORT" 2>/dev/null; then
        log_success "Minecraft server is running on $MC_HOST:$MC_PORT"
        return 0
    else
        log_error "Cannot connect to Minecraft server at $MC_HOST:$MC_PORT"
        return 1
    fi
}

# Function to wait for port to be available
wait_for_port() {
    local port=$1
    local timeout=10
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if timeout 1 bash -c "echo > /dev/tcp/127.0.0.1/$port" 2>/dev/null; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    
    return 1
}

# Main test execution
main() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════╗"
    echo "║     TFG_Chantada Agent Test Suite      ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Check prerequisites
    log_info "Checking prerequisites..."
    
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed"
        exit 1
    fi
    log_success "Node.js found: $(node --version)"
    
    # Check if running from correct directory
    if [ ! -f "package.json" ]; then
        log_error "package.json not found. Run from project root directory"
        exit 1
    fi
    
    # Check Minecraft server
    if ! check_minecraft; then
        log_warning "Minecraft server is not accessible"
        log_info "Make sure Minecraft is running and world is open to LAN on $MC_HOST:$MC_PORT"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Cleanup metrics if requested
    if [ "$CLEANUP_METRICS" = true ]; then
        cleanup_metrics
    fi
    
    echo ""
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}Test 1: Single HTN Agent${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo ""
    log_info "Launching MindServer on port $MINDSERVER_PORT..."
    
    # Start MindServer in background
    timeout 120 node src/server/mindcraft.js \
        --port "$MINDSERVER_PORT" \
        --minecraft-host "$MC_HOST" \
        --minecraft-port "$MC_PORT" &
    
    MINDSERVER_PID=$!
    log_info "MindServer PID: $MINDSERVER_PID"
    
    # Wait for MindServer to start
    log_info "Waiting for MindServer to start..."
    if ! wait_for_port "$MINDSERVER_PORT"; then
        log_error "MindServer failed to start"
        kill $MINDSERVER_PID 2>/dev/null || true
        exit 1
    fi
    log_success "MindServer started"
    
    sleep 2
    
    # Launch HTN Agent
    echo ""
    log_info "Launching HTN Agent (Agent1)..."
    log_info "Command: node src/agents/add_agent.js --name Agent1 --type htn --port $MINDSERVER_PORT -c 0"
    echo ""
    
    timeout 300 node src/agents/add_agent.js \
        --name Agent1 \
        --type htn \
        --port "$MINDSERVER_PORT" \
        --minecraft-port "$MC_PORT" \
        -c 0 || true
    
    AGENT_EXIT_CODE=$?
    
    # Cleanup
    log_info "Shutting down MindServer..."
    kill $MINDSERVER_PID 2>/dev/null || true
    wait $MINDSERVER_PID 2>/dev/null || true
    
    echo ""
    if [ $AGENT_EXIT_CODE -eq 0 ] || [ $AGENT_EXIT_CODE -eq 124 ]; then
        log_success "Test completed (Agent finished or timeout reached)"
    else
        log_error "Test failed with exit code $AGENT_EXIT_CODE"
        exit 1
    fi
}

# Run main function
main

log_success "All tests completed!"
echo ""
log_info "Tips for running agents:"
echo "  - Single agent: node src/agents/add_agent.js --name MyAgent --type htn"
echo "  - Multiple agents: Launch add_agent.js in separate terminals"
echo "  - View metrics: Check src/metrics/agent_metrics/ directory"
echo ""

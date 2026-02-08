@echo off
REM Test script for quickly launching and testing agents
REM Requires a Minecraft server running on localhost:25565

setlocal enabledelayedexpansion

REM Configuration
set "MC_HOST=127.0.0.1"
set "MC_PORT=25565"
set "MINDSERVER_PORT=8080"
set "CLEANUP_METRICS=false"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--help" goto show_help
if "%~1"=="-h" goto show_help
if "%~1"=="--cleanup" (
    set "CLEANUP_METRICS=true"
    shift
    goto parse_args
)
if "%~1"=="-h" (
    set "MC_HOST=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--host" (
    set "MC_HOST=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="-p" (
    set "MC_PORT=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--minecraft-port" (
    set "MC_PORT=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--mindserver-port" (
    set "MINDSERVER_PORT=%~2"
    shift
    shift
    goto parse_args
)

echo Unknown option: %~1
exit /b 1

:show_help
echo Usage: test_agents.bat [OPTIONS]
echo.
echo Options:
echo   -h, --host MC_HOST              Minecraft server host (default: 127.0.0.1)
echo   -p, --minecraft-port PORT       Minecraft server port (default: 25565)
echo   --mindserver-port PORT          MindServer port (default: 8080)
echo   --cleanup                       Clean metrics directory before running
echo   --help, -h                      Show this help message
echo.
echo Examples:
echo   test_agents.bat                          - Use defaults
echo   test_agents.bat --cleanup                - Clean metrics and run
echo   test_agents.bat -p 25566 --cleanup       - Custom MC port + cleanup
echo.
exit /b 0

:args_done
cls
echo.
echo ╔════════════════════════════════════════╗
echo ║     TFG_Chantada Agent Test Suite      ║
echo ╚════════════════════════════════════════╝
echo.

REM Check prerequisites
echo [INFO] Checking prerequisites...

where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js is not installed
    exit /b 1
)

for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
echo [SUCCESS] Node.js found: %NODE_VERSION%

REM Check if running from correct directory
if not exist "package.json" (
    echo [ERROR] package.json not found. Run from project root directory
    exit /b 1
)

REM Cleanup metrics if requested
if "%CLEANUP_METRICS%"=="true" (
    echo [INFO] Cleaning metrics directory...
    if exist "src\metrics\agent_metrics" (
        cd "src\metrics\agent_metrics"
        for /f "delims=" %%F in ('dir /b ^| findstr /v "example_"') do (
            del /q "%%F" 2>nul
        )
        cd "..\..\..\"
        echo [SUCCESS] Metrics cleaned
    )
)

echo.
echo ════════════════════════════════════════
echo Test 1: Single HTN Agent
echo ════════════════════════════════════════
echo.

echo [INFO] Launching MindServer on port %MINDSERVER_PORT%...
REM Start MindServer in a new window
start "MindServer" cmd /k "node src\server\mindcraft.js --port %MINDSERVER_PORT% --minecraft-host %MC_HOST% --minecraft-port %MC_PORT%"

REM Wait for MindServer to start
timeout /t 3 /nobreak

echo [INFO] MindServer started
echo.

echo [INFO] Launching HTN Agent (Agent1)...
echo [INFO] Command: node src\agents\add_agent.js --name Agent1 --type htn --port %MINDSERVER_PORT% -c 0
echo.

node src\agents\add_agent.js ^
    --name Agent1 ^
    --type htn ^
    --port %MINDSERVER_PORT% ^
    --minecraft-port %MC_PORT% ^
    -c 0

echo.
echo [SUCCESS] Test completed!
echo.
echo Tips for running agents:
echo   - Single agent: node src\agents\add_agent.js --name MyAgent --type htn
echo   - Multiple agents: Launch add_agent.js in separate terminals
echo   - View metrics: Check src\metrics\agent_metrics\ directory
echo.

pause

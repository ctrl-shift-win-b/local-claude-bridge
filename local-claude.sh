#!/usr/bin/env bash
# Linux launcher: starts the llama.cpp server, the bridge, then Claude Code.
# Mirrors local-claude.ps1. Usage: ./local-claude.sh [--debug-bridge] [--no-poke]
#                                    [--vision-internal] [--vision-external <url>]
#                                    [--agents-url <url>] [--no-agents] [claude args...]

set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEBUG_BRIDGE=0
NO_POKE=1
VISION_INTERNAL=0
VISION_EXTERNAL=""
AGENTS_URL=""
NO_AGENTS=0
PASSTHRU=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug-bridge)    DEBUG_BRIDGE=1; shift ;;
        --no-poke)         NO_POKE=1; shift ;;
        --vision-internal) VISION_INTERNAL=1; shift ;;
        --vision-external) VISION_EXTERNAL="$2"; shift 2 ;;
        --agents-url)      AGENTS_URL="$2"; shift 2 ;;
        --no-agents)       NO_AGENTS=1; shift ;;
        *)                 PASSTHRU+=("$1"); shift ;;
    esac
done

kill_port() {
    local port="$1"
    local pids
    pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        kill -9 $pids 2>/dev/null || true
    fi
}

wait_health() {
    local url="$1"
    local interval="$2"
    while true; do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "$url" 2>/dev/null || echo "000")
        [[ "$code" == "200" ]] && return
        sleep "$interval"
    done
}

kill_port 1235
kill_port 1234

# Must match start_server.sh's CONTEXT_SIZE default. n_ctx stays at the native
# 256K so a large tool result on top of compacted history still fits; Claude
# Code is capped at 70% so autocompact/prefill stay bounded. Exported so an
# override here also reaches start_server.sh (run as a child process below).
export CONTEXT_SIZE="${CONTEXT_SIZE:-262144}"
MAX_CONTEXT_TOKENS=$(( CONTEXT_SIZE * 70 / 100 ))

SERVER_PID=""
BRIDGE_PID=""

cleanup() {
    echo ""
    echo "Shutting down..."
    [[ -n "$BRIDGE_PID" ]] && kill "$BRIDGE_PID" 2>/dev/null
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    kill_port 1235
    kill_port 1234
    echo "Done."
}
trap cleanup EXIT

LLAMA_LOG="$DIR/llama-server.log"
BRIDGE_LOG="$DIR/bridge-stdout.log"

"$DIR/start_server.sh" > "$LLAMA_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for llama.cpp server... (log: $LLAMA_LOG)"
wait_health "http://localhost:1234/health" 2
echo "llama.cpp ready."

BRIDGE_ARGS=("$DIR/bridge.py")
[[ "$DEBUG_BRIDGE"    -eq 1 ]] && BRIDGE_ARGS+=(--debug)
[[ "$NO_POKE"         -eq 1 ]] && BRIDGE_ARGS+=(--no-poke)
[[ "$VISION_INTERNAL" -eq 1 ]] && BRIDGE_ARGS+=(--image-processing-internal)
[[ -n "$VISION_EXTERNAL" ]]    && BRIDGE_ARGS+=(--image-processing-external "$VISION_EXTERNAL")
[[ -n "$AGENTS_URL" ]]         && BRIDGE_ARGS+=(--agents-url "$AGENTS_URL")
[[ "$NO_AGENTS"       -eq 1 ]] && BRIDGE_ARGS+=(--no-agents)

"$DIR/.venv/bin/python" "${BRIDGE_ARGS[@]}" > "$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!

echo "Waiting for bridge... (log: $BRIDGE_LOG)"
wait_health "http://localhost:1235/health" 1
echo "Bridge ready."

export ANTHROPIC_BASE_URL="http://localhost:1235"
export ANTHROPIC_AUTH_TOKEN="local"
export ANTHROPIC_MODEL="local-model"
export CLAUDE_CODE_ATTRIBUTION_HEADER="0"
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE="75"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$MAX_CONTEXT_TOKENS"

# llama.cpp --parallel 1: a subagent is a second full context that evicts the
# main KV cache and re-prefills. Hide Agent unless ALLOW_AGENTS=1. Concurrent
# cannot be 0 (Claude Code ignores non-positive values); 1 is the floor.
export CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS="${CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS:-1}"
export CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH="${CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH:-1}"
export CLAUDE_CODE_DISABLE_EXPLORE_PLAN_AGENTS="${CLAUDE_CODE_DISABLE_EXPLORE_PLAN_AGENTS:-1}"
export CLAUDE_CODE_FORK_SUBAGENT="${CLAUDE_CODE_FORK_SUBAGENT:-0}"

CLAUDE_ARGS=()
if [[ "${ALLOW_AGENTS:-0}" != "1" ]]; then
    CLAUDE_ARGS+=(--disallowedTools Agent)
fi
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
    CLAUDE_ARGS+=("${PASSTHRU[@]}")
fi
claude "${CLAUDE_ARGS[@]}"

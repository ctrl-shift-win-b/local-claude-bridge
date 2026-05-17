#!/usr/bin/env bash
# Connect a remote Claude Code session to an existing bridge + llama.cpp server.
# The server and bridge must already be running on the host machine.
#
# Usage: ./remote-claude.sh <server_ip> <bridge_port> [claude args...]
#   server_ip    IP of the Windows host running the bridge
#   bridge_port  Port the bridge is listening on (typically 1235)
#   claude args  Any extra arguments are passed directly to claude

set -euo pipefail

usage() {
    echo "Usage: $0 <server_ip> <bridge_port> [claude args...]"
    echo ""
    echo "  server_ip    IP address of the host running the bridge"
    echo "  bridge_port  Port the bridge is listening on (typically 1235)"
    echo ""
    echo "Examples:"
    echo "  $0 192.168.1.50 1235"
    echo "  $0 192.168.1.50 1235 --continue"
    echo "  $0 192.168.1.50 1235 --debug"
    exit 1
}

[[ $# -lt 2 ]] && usage

SERVER_IP="$1"
BRIDGE_PORT="$2"
shift 2

export ANTHROPIC_BASE_URL="http://${SERVER_IP}:${BRIDGE_PORT}"
export ANTHROPIC_AUTH_TOKEN="local"
export ANTHROPIC_MODEL="local-model"
export CLAUDE_CODE_ATTRIBUTION_HEADER="0"
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE="75"

echo "Bridge:  ${ANTHROPIC_BASE_URL}"

# Verify bridge is reachable before handing off to claude
if ! curl -sf --max-time 3 "${ANTHROPIC_BASE_URL}/health" > /dev/null; then
    echo ""
    echo "ERROR: bridge not reachable at ${ANTHROPIC_BASE_URL}" >&2
    echo "Make sure the Windows host is running start_server.bat and the bridge." >&2
    exit 1
fi

echo "Bridge OK. Starting Claude Code..."
echo ""

exec claude "$@"

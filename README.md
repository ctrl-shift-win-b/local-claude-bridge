# local-claude-bridge

An Anthropic Messages API ↔ OpenAI Chat Completions bridge that allows [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) to communicate with a local [llama.cpp](https://github.com/ggml-org/llama.cpp) server (e.g., llama.cpp, LM Studio, Ollama).

## What it does

`bridge.py` translates between Claude Code's Anthropic API and any OpenAI-compatible server, running in-process so the local model is a drop-in replacement.

## Why the poke mechanism?

A bare chat template pass-through has one critical weakness: when a local model is asked to use tools, it often **thinks** about calling a tool but forgets to emit the tool call, ending with `stop_reason: end_turn` and no `tool_use` block. The result is silent failure — Claude Code sees an empty reply and moves on.

The bridge's **poke mechanism** detects this pattern and pushes the model to complete its intended action:

- Detects when the model's think block mentions a tool name it was supposed to call
- Falls back to trigger phrases like "Let me call..." or "I'll use..." in the output text
- On hard stalls (empty response immediately after a tool result), pokes unconditionally
- Retries up to **2 times** with a 1-second delay between attempts — the delay ensures client disconnects cancel the poke cleanly without wasted work
- When plan mode is active, suppresses poke for write, edit, and agent-style tool calls to avoid misfiring on responses that are intentionally final in that context

This turns a local model that *almost* does tool calling correctly into one that actually does, without any prompt engineering or template changes.

## Features

| Feature | Description |
|---|---|
| **Full request translation** | Anthropic system messages, multi-part content blocks, tool results, and scalar params → OpenAI format and back |
| **Poke/continuation** | Detects abandoned tool calls from think-block analysis and trigger phrases, re-prompts the model automatically |
| **Schema-aware parameter coercion** | Fixes common local model output errors — array-to-string, enum mismatches via substring matching and alias table |
| **Streaming with poke** | Buffered streaming for tool-calling paths; real-time SSE streaming when no tools are involved |
| **Think block stripping** | `<think>...</think>` content is transparently removed from responses sent to Claude Code |
| **Usage tracking** | `stream_options: {"include_usage": True}` on all streaming requests, output tokens propagated back to Claude Code |
| **Local web search** | `web_search` and `web_fetch` tool calls are executed by the bridge via DuckDuckGo — no Anthropic API required |
| **Native tool schema injection** | Anthropic native tool types (`web_search_20250305`, `web_fetch_*`) carry no `input_schema`; the bridge injects proper parameter schemas so the local model knows what args to emit |
| **External vision** | Images pasted inline or returned by the `Read` tool are routed to a separate vision model; descriptions are injected in-place so the main model sees them as text |
| **Vision cache** | Per-process SHA-256 cache prevents re-analyzing the same image on every turn |
| **Debug mode** | Pass `--debug` to `bridge.py` to log every request body, response body, and SSE chunk to `bridge.log` |

## Setup

### Dependencies

```bat
.venv\Scripts\pip install fastapi uvicorn httpx ddgs
```

### Launching

```bat
.\start_local_claude.ps1
```

This starts the llama.cpp server, waits for it to be healthy, starts the bridge on `localhost:1235`, then launches Claude Code pointed at the bridge.

To enable full debug logging:

```bat
.\start_local_claude.ps1 -DebugBridge
```

Everything — request bodies, response bodies, every SSE chunk — is written to `bridge.log`.

## Configuration

Environment variables set by `start_local_claude.ps1`:

| Variable | Value |
|---|---|
| `ANTHROPIC_BASE_URL` | `http://localhost:1235` |
| `ANTHROPIC_AUTH_TOKEN` | `local` |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | `0` |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | `50` |

Bridge configuration constants in `bridge.py`:

| Variable | Description | Default |
|---|---|---|
| `BRIDGE_PORT` | Bridge listening port | `1235` |
| `LLAMA_BASE_URL` | Upstream llama.cpp server | `http://localhost:1234` |
| `POKE_ENABLED` | Enable poke/continuation mechanism | `True` |
| `POKE_MAX_RETRIES` | Max poke attempts per request | `2` |
| `POKE_DELAY_SECONDS` | Delay before poke (client disconnect cancels) | `1.0` |
| `BRIDGE_TOOL_MAX_ITER` | Max bridge tool execution iterations | `8` |
| `BRIDGE_VISION_MAX_TOKENS` | Max tokens for vision model description | `2048` |
| `BRIDGE_VISION_MODEL` | Vision model ID sent to the vision server | `qwen/qwen3-vl-4b` |
| `BRIDGE_VISION_TEMPERATURE` | Vision model temperature | `0.3` |
| `BRIDGE_VISION_TIMEOUT` | Per-image vision request timeout (seconds) | `60.0` |

## Image / Vision Support

The bridge has three vision modes, selected at startup:

| Mode | Flag | Behavior |
|---|---|---|
| `disabled` (default) | _(none)_ | Image blocks are stripped and replaced with a "model cannot see images" notice |
| `internal` | `--image-processing-internal` | Images are passed through to the main llama.cpp server as-is (requires a multimodal model) |
| `external` | `--image-processing-external <url>` | Images are sent to a separate vision server; descriptions are injected as `[Image:]` text blocks |

### External mode

Start the bridge with the vision server URL:

```bat
python bridge.py --image-processing-external http://localhost:1234/v1/chat/completions
```

What happens per request:

1. Every `image` content block in the full message history is found — including blocks nested inside `tool_result` messages (e.g. from Claude Code's `Read` tool used on an image file).
2. All images in a single message are resolved in **parallel** via `asyncio.gather`.
3. Each image block is **replaced in-place** with a `[Image:\n{description}\n]` text block before the request is forwarded to the main model.
4. Descriptions are cached by SHA-256 of the image data for the lifetime of the bridge process — the same image appearing in subsequent turns costs zero additional vision calls.

The system prompt is automatically extended with a relay instruction that tells the main model to describe what it "sees" from the `[Image:]` blocks without acknowledging the preprocessing mechanism.

## Local Web Search

Claude Code's `WebSearch` tool routes through the bridge. When Qwen3 calls `web_search`, the bridge executes the query via DuckDuckGo (`ddgs`) and returns the results directly as plain text to Claude Code. No Anthropic API key or network calls to Anthropic required.

The bridge also handles `web_fetch` — it GETs the URL, strips navigation/scripts/ads from the HTML, and returns up to 12 000 characters of readable text.

## Known Issues

### Web search reports "Did 0 searches in Xs"

Claude Code's search counter increments by counting `web_search_result_block` content blocks in the haiku response. The bridge returns search results as plain text instead (the only format Claude Code actually forwards to the main model — `encrypted_content` in `web_search_result_block` is Anthropic server-side only and is silently discarded when it comes from a third-party endpoint). The counter will always show 0; the search results themselves are real and delivered correctly.

### Occasional stalls requiring manual "continue"

The poke mechanism catches the most common failure mode — model mentions a tool in its think block but doesn't emit the call — but it has limits:

- The model occasionally produces a response that looks complete to the bridge but is actually mid-thought, and the poke heuristics don't fire. Typing "continue" or repeating the request manually recovers it.
- After `POKE_MAX_RETRIES` (default 2) attempts the bridge gives up regardless of outcome. A model that needs more than two nudges will stall.
- The hard-stall detector (empty response after a tool result) only fires once. If the model stalls twice in the same turn the second stall is not caught.

### False positive pokes

The poke trigger phrases ("I'll call...", "Let me use...", etc.) and the think-block tool-name check occasionally fire on responses where the model was legitimately done — for example, when it writes about a tool in an explanatory context rather than intending to call it. This causes the bridge to re-prompt unnecessarily, which either produces a duplicate or an unwanted tool call. The list of trigger phrases in `POKE_TRIGGER_PHRASES` can be tightened if false positives are frequent for a specific model.

## Files

| File | Description |
|---|---|
| `bridge.py` | Main bridge server (FastAPI) |
| `start_server.bat` | Start llama.cpp server |
| `start_local_claude.ps1` | PowerShell launcher — starts server, bridge, and Claude Code |
| `tests/test_bridge.py` | 82-test pytest suite |
| `bridge.log` | Server log (created at runtime) |
| `LICENSE` | MIT License |

## License

[MIT](LICENSE)

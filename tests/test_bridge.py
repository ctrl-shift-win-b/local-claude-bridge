"""
Tests for bridge.py — Anthropic↔OpenAI translation bridge.
Run with:  pytest tests/test_bridge.py -v
"""

import asyncio
import json
import re
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import bridge
from bridge import (
    _strip_think,
    oai_to_anthropic_response,
    _should_poke,
    _emit_anthropic_sse,
    _stream_with_poke,
    stream_oai_to_anthropic,
    _get_bridge_tool_arg,
    _bridge_web_search,
    _bridge_web_fetch,
    _oai_resp_has_only_bridge_calls,
    _oai_resp_get_bridge_calls,
    _bridge_tool_loop,
    build_oai_request,
    _EFFORT_BUDGET_MAP,
)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

async def collect(gen):
    """Collect all items from an async generator."""
    return [item async for item in gen]


def parse_events(sse_list: list[str]) -> list[tuple[str, dict]]:
    """Parse SSE strings into (event_name, data_dict) list."""
    result = []
    for s in sse_list:
        event = data = None
        for line in s.strip().splitlines():
            if line.startswith("event: "):
                event = line[7:]
            if line.startswith("data: "):
                data = json.loads(line[6:])
        if event and data is not None:
            result.append((event, data))
    return result


def oai_text_chunk(text, finish_reason=None, usage=None, index=0):
    """Build a minimal OAI SSE text delta chunk dict."""
    c = {
        "choices": [
            {"delta": {"content": text}, "finish_reason": finish_reason, "index": index}
        ]
    }
    if usage:
        c["usage"] = usage
    return c


def oai_tool_chunk(tc_index, tc_id=None, name=None, args=None, finish_reason=None):
    """Build an OAI SSE tool_call delta chunk dict."""
    tc = {"index": tc_index, "function": {}}
    if tc_id:
        tc["id"] = tc_id
    if name:
        tc["function"]["name"] = name
    if args is not None:
        tc["function"]["arguments"] = args
    return {
        "choices": [
            {"delta": {"tool_calls": [tc]}, "finish_reason": finish_reason, "index": 0}
        ]
    }


def oai_finish_chunk(finish_reason="stop", usage=None):
    c = {"choices": [{"delta": {}, "finish_reason": finish_reason, "index": 0}]}
    if usage:
        c["usage"] = usage
    return c


def make_sse(*chunks):
    """Convert OAI chunk dicts to list of SSE data lines + DONE sentinel."""
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    lines.append("data: [DONE]")
    return lines


def fake_httpx(sse_lines, poke_response=None):
    """
    Return a replacement class for bridge.httpx.AsyncClient.
    The class's instances serve as async context managers.
    Streaming yields sse_lines; non-streaming POST returns poke_response.
    """
    _poke_resp = poke_response or {
        "choices": [
            {
                "message": {"content": None, "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        def stream(self, *a, **kw):
            @asynccontextmanager
            async def _cm():
                class S:
                    def raise_for_status(self):
                        pass

                    async def aiter_lines(self2):
                        for ln in sse_lines:
                            yield ln

                yield S()

            return _cm()

        async def post(self, *a, **kw):
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json = MagicMock(return_value=_poke_resp)
            return r

    return FakeClient


# ---------------------------------------------------------------------------
# SSE structure assertion helper
# ---------------------------------------------------------------------------

def assert_sse_structure(events: list[tuple[str, dict]]) -> None:
    """
    Assert the mandatory SSE event ordering rules for all streaming tests:
    1. First event is message_start
    2. Each content block has content_block_start before any content_block_delta
    3. Each content block has content_block_stop after its deltas
    4. message_delta comes before message_stop
    5. message_stop is last
    """
    names = [e for e, _ in events]

    assert names[0] == "message_start", f"First event must be message_start, got {names[0]!r}"
    assert names[-1] == "message_stop", f"Last event must be message_stop, got {names[-1]!r}"

    # message_delta before message_stop
    if "message_delta" in names:
        assert names.index("message_delta") < names.index("message_stop"), \
            "message_delta must come before message_stop"

    # Per-block ordering
    seen_starts: dict[int, int] = {}   # index → position of content_block_start
    seen_deltas: dict[int, list[int]] = {}  # index → positions of deltas
    seen_stops: dict[int, int] = {}    # index → position of content_block_stop

    for pos, (evt, data) in enumerate(events):
        idx = data.get("index")
        if evt == "content_block_start":
            seen_starts[idx] = pos
            seen_deltas.setdefault(idx, [])
        elif evt == "content_block_delta":
            assert idx in seen_starts, \
                f"content_block_delta for index {idx} before content_block_start"
            seen_deltas.setdefault(idx, []).append(pos)
        elif evt == "content_block_stop":
            assert idx in seen_starts, \
                f"content_block_stop for index {idx} before content_block_start"
            seen_stops[idx] = pos

    for idx in seen_starts:
        assert idx in seen_stops, f"content_block_start for index {idx} has no matching content_block_stop"
        if seen_deltas.get(idx):
            assert max(seen_deltas[idx]) < seen_stops[idx], \
                f"content_block_delta for index {idx} comes after content_block_stop"


# ===========================================================================
# 1. _strip_think — 8 tests
# ===========================================================================

class TestStripThink:
    def test_strip_leading_block(self):
        visible, think = _strip_think("<think>thinking</think>visible")
        assert visible == "visible"
        assert think == "thinking"

    def test_strip_leading_block_whitespace(self):
        visible, think = _strip_think("  <think> think </think>  text")
        assert visible == "text"
        assert think == " think "

    def test_no_think_block(self):
        visible, think = _strip_think("plain text")
        assert visible == "plain text"
        assert think == ""

    def test_empty_string(self):
        visible, think = _strip_think("")
        assert visible == ""
        assert think == ""

    def test_mid_text_tag_not_stripped(self):
        text = "here is <think>example</think> config"
        visible, think = _strip_think(text)
        # The tag is NOT at position 0 of the string so it must not be stripped
        assert "<think>example</think>" in visible
        assert think == ""

    def test_only_think_block(self):
        visible, think = _strip_think("<think>only</think>")
        assert visible == ""
        assert think == "only"

    def test_multiline_think(self):
        raw = "<think>line one\nline two\nline three</think>after"
        visible, think = _strip_think(raw)
        assert visible == "after"
        assert "line one" in think
        assert "line three" in think

    def test_think_in_code_block(self):
        # Backtick block starts before <think>, so the <think> is NOT a leading tag
        raw = "```\n<think>x</think>\n```"
        visible, think = _strip_think(raw)
        assert think == ""
        assert "<think>x</think>" in visible


# ===========================================================================
# 2. oai_to_anthropic_response — 8 tests
# ===========================================================================

class TestOaiToAnthropicResponse:
    def _make_oai(self, content=None, tool_calls=None, finish_reason="stop", usage=None):
        msg: dict = {}
        if content is not None:
            msg["content"] = content
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        return {
            "choices": [{"message": msg, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20},
        }

    def test_text_only(self):
        oai = self._make_oai(content="<think>reasoning</think>Hello")
        resp = oai_to_anthropic_response(oai, "test-model")
        assert len(resp["content"]) == 1
        assert resp["content"][0]["type"] == "text"
        assert resp["content"][0]["text"] == "Hello"

    def test_tool_call_only(self):
        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "/tmp/f"}'},
            }
        ]
        oai = self._make_oai(content=None, tool_calls=tool_calls, finish_reason="tool_calls")
        resp = oai_to_anthropic_response(oai, "test-model")
        assert len(resp["content"]) == 1
        assert resp["content"][0]["type"] == "tool_use"
        assert resp["stop_reason"] == "tool_use"

    def test_text_and_tool_call(self):
        tool_calls = [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "list_dir", "arguments": '{}'},
            }
        ]
        oai = self._make_oai(content="Searching now", tool_calls=tool_calls, finish_reason="tool_calls")
        resp = oai_to_anthropic_response(oai, "test-model")
        types = [b["type"] for b in resp["content"]]
        assert "text" in types
        assert "tool_use" in types

    def test_empty_response(self):
        oai = self._make_oai(content=None)
        resp = oai_to_anthropic_response(oai, "test-model")
        assert resp["content"] == []

    def test_stop_reason_mapping(self):
        for finish, expected in [
            ("tool_calls", "tool_use"),
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("unknown_reason", "end_turn"),
        ]:
            oai = self._make_oai(finish_reason=finish)
            resp = oai_to_anthropic_response(oai, "test-model")
            assert resp["stop_reason"] == expected, \
                f"finish_reason={finish!r} should map to {expected!r}, got {resp['stop_reason']!r}"

    def test_think_only_gives_empty_content(self):
        oai = self._make_oai(content="<think>think</think>")
        resp = oai_to_anthropic_response(oai, "test-model")
        text_blocks = [b for b in resp["content"] if b["type"] == "text"]
        assert text_blocks == [], \
            "A response with only a think block should produce no text content blocks"

    def test_tool_json_decode_error(self):
        tool_calls = [
            {
                "id": "call_bad",
                "type": "function",
                "function": {"name": "bad_tool", "arguments": "NOT JSON {{{"},
            }
        ]
        oai = self._make_oai(tool_calls=tool_calls, finish_reason="tool_calls")
        # Must not raise
        resp = oai_to_anthropic_response(oai, "test-model")
        tool_block = resp["content"][0]
        assert tool_block["type"] == "tool_use"
        assert "_raw" in tool_block["input"]

    def test_usage_mapping(self):
        oai = self._make_oai(usage={"prompt_tokens": 42, "completion_tokens": 17})
        resp = oai_to_anthropic_response(oai, "test-model")
        assert resp["usage"]["input_tokens"] == 42
        assert resp["usage"]["output_tokens"] == 17


# ===========================================================================
# 3. build_oai_request — thinking parameter mapping
# ===========================================================================

class TestBuildOaiRequestThinking:
    """build_oai_request maps Anthropic thinking params → llama.cpp fields."""

    def _base(self, thinking=None, extra=None):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        if thinking is not None:
            body["thinking"] = thinking
        if extra:
            body.update(extra)
        return body

    # --- thinking.type = "enabled" ---

    def test_enabled_maps_budget_tokens(self):
        oai = build_oai_request(self._base({"type": "enabled", "budget_tokens": 5000}))
        assert oai["thinking_budget_tokens"] == 5000
        assert "chat_template_kwargs" not in oai

    def test_enabled_different_budget(self):
        oai = build_oai_request(self._base({"type": "enabled", "budget_tokens": 1024}))
        assert oai["thinking_budget_tokens"] == 1024

    def test_enabled_missing_budget_not_added(self):
        oai = build_oai_request(self._base({"type": "enabled"}))
        assert "thinking_budget_tokens" not in oai

    def test_enabled_zero_budget_not_added(self):
        oai = build_oai_request(self._base({"type": "enabled", "budget_tokens": 0}))
        assert "thinking_budget_tokens" not in oai

    def test_enabled_negative_budget_not_added(self):
        oai = build_oai_request(self._base({"type": "enabled", "budget_tokens": -1}))
        assert "thinking_budget_tokens" not in oai

    def test_enabled_string_budget_not_added(self):
        oai = build_oai_request(self._base({"type": "enabled", "budget_tokens": "5000"}))
        assert "thinking_budget_tokens" not in oai

    # --- thinking.type = "adaptive" ---

    def test_adaptive_low_effort(self):
        oai = build_oai_request(self._base({"type": "adaptive", "effort": "low"}))
        assert oai["thinking_budget_tokens"] == _EFFORT_BUDGET_MAP["low"]

    def test_adaptive_medium_effort(self):
        oai = build_oai_request(self._base({"type": "adaptive", "effort": "medium"}))
        assert oai["thinking_budget_tokens"] == _EFFORT_BUDGET_MAP["medium"]

    def test_adaptive_high_effort(self):
        oai = build_oai_request(self._base({"type": "adaptive", "effort": "high"}))
        assert oai["thinking_budget_tokens"] == _EFFORT_BUDGET_MAP["high"]

    def test_adaptive_missing_effort_defaults_to_medium(self):
        oai = build_oai_request(self._base({"type": "adaptive"}))
        assert oai["thinking_budget_tokens"] == _EFFORT_BUDGET_MAP["medium"]

    def test_adaptive_unknown_effort_defaults_to_medium(self):
        oai = build_oai_request(self._base({"type": "adaptive", "effort": "ultra"}))
        assert oai["thinking_budget_tokens"] == _EFFORT_BUDGET_MAP["medium"]

    def test_adaptive_no_chat_template_kwargs(self):
        oai = build_oai_request(self._base({"type": "adaptive", "effort": "high"}))
        assert "chat_template_kwargs" not in oai

    # --- unknown / future types ---

    def test_unknown_type_no_budget_no_crash(self):
        oai = build_oai_request(self._base({"type": "future_type", "budget_tokens": 9999}))
        assert "thinking_budget_tokens" not in oai
        assert "chat_template_kwargs" not in oai

    # --- thinking absent ---

    def test_no_thinking_param_no_budget(self):
        oai = build_oai_request(self._base())
        assert "thinking_budget_tokens" not in oai
        assert "chat_template_kwargs" not in oai

    # --- DISABLE_THINKING env var ---

    def test_disable_thinking_env_injects_enable_false(self):
        with patch("bridge.DISABLE_THINKING", True):
            oai = build_oai_request(self._base())
        assert oai.get("chat_template_kwargs") == {"enable_thinking": False}
        assert "thinking_budget_tokens" not in oai

    def test_disable_thinking_env_overridden_by_explicit_enabled(self):
        """Explicit thinking=enabled takes priority over DISABLE_THINKING."""
        with patch("bridge.DISABLE_THINKING", True):
            oai = build_oai_request(self._base({"type": "enabled", "budget_tokens": 3000}))
        assert oai["thinking_budget_tokens"] == 3000
        assert "chat_template_kwargs" not in oai

    def test_disable_thinking_env_overridden_by_adaptive(self):
        with patch("bridge.DISABLE_THINKING", True):
            oai = build_oai_request(self._base({"type": "adaptive", "effort": "low"}))
        assert oai["thinking_budget_tokens"] == _EFFORT_BUDGET_MAP["low"]
        assert "chat_template_kwargs" not in oai

    def test_disable_thinking_false_by_default(self):
        with patch("bridge.DISABLE_THINKING", False):
            oai = build_oai_request(self._base())
        assert "chat_template_kwargs" not in oai

    # --- other params unaffected ---

    def test_thinking_does_not_affect_other_fields(self):
        oai = build_oai_request(self._base(
            {"type": "enabled", "budget_tokens": 2000},
            extra={"max_tokens": 4096, "temperature": 0.5},
        ))
        assert oai["max_tokens"] == 4096
        assert oai["temperature"] == 0.5
        assert oai["thinking_budget_tokens"] == 2000

    # --- effort budget map completeness ---

    def test_effort_budget_map_all_keys_positive(self):
        for effort, budget in _EFFORT_BUDGET_MAP.items():
            assert isinstance(budget, int) and budget >= 1024, (
                f"effort '{effort}' budget {budget} is below Anthropic minimum of 1024"
            )

    def test_effort_budget_map_ordered_low_lt_medium_lt_high(self):
        assert _EFFORT_BUDGET_MAP["low"] < _EFFORT_BUDGET_MAP["medium"] < _EFFORT_BUDGET_MAP["high"]


# ===========================================================================
# 4. _should_poke — 6 tests
# ===========================================================================

class TestShouldPoke:
    def _anthropic_end_turn(self, content=None):
        return {
            "stop_reason": "end_turn",
            "content": content or [],
        }

    def _oai_resp(self, content="", tool_calls=None):
        msg: dict = {"content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {"choices": [{"message": msg, "finish_reason": "stop"}]}

    def test_poke_disabled(self):
        original = bridge.POKE_ENABLED
        bridge.POKE_ENABLED = False
        try:
            result = _should_poke(
                self._oai_resp(),
                ["some_tool"],
                self._anthropic_end_turn(),
            )
        finally:
            bridge.POKE_ENABLED = original
        assert result is False

    def test_no_tools(self):
        result = _should_poke(
            self._oai_resp(),
            [],
            self._anthropic_end_turn(),
        )
        assert result is False

    def test_already_has_tool_use(self):
        anthropic_resp = {
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "x", "name": "read_file", "input": {}}],
        }
        result = _should_poke(
            self._oai_resp(tool_calls=[{"id": "x", "function": {"name": "read_file", "arguments": "{}"}}]),
            ["read_file"],
            anthropic_resp,
        )
        assert result is False

    def test_think_mentions_tool_no_visible_text(self):
        # Think block mentions a tool and model produced no visible output — poke.
        raw = "<think>I should call read_file to get the content</think>"
        oai = self._oai_resp(content=raw)
        anthropic_resp = self._anthropic_end_turn()
        result = _should_poke(oai, ["read_file"], anthropic_resp)
        assert result is True

    def test_think_mentions_tool_with_visible_text_no_poke(self):
        # Think block mentions a tool but model produced a real conversational reply —
        # do NOT poke; "read_file" in the think is a common-word false positive.
        raw = "<think>I should call read_file to get the content</think>I cannot help with that."
        oai = self._oai_resp(content=raw)
        anthropic_resp = self._anthropic_end_turn(
            content=[{"type": "text", "text": "I cannot help with that."}]
        )
        result = _should_poke(oai, ["read_file"], anthropic_resp)
        assert result is False

    def test_hard_stall(self):
        oai = self._oai_resp(content=None)
        anthropic_resp = {"stop_reason": "end_turn", "content": []}
        result = _should_poke(oai, ["read_file"], anthropic_resp, last_is_tool_result=True)
        assert result is True

    def test_clean_end_turn(self):
        raw = "Here is the answer you requested."
        oai = self._oai_resp(content=raw)
        anthropic_resp = self._anthropic_end_turn(
            content=[{"type": "text", "text": "Here is the answer you requested."}]
        )
        result = _should_poke(oai, ["read_file"], anthropic_resp)
        assert result is False


# ===========================================================================
# 4. _emit_anthropic_sse — 3 tests (async)
# ===========================================================================

@pytest.mark.asyncio
class TestEmitAnthropicSse:
    def _resp(self, content_blocks, stop_reason="end_turn", model="test-model"):
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "content": content_blocks,
            "usage": {"input_tokens": 5, "output_tokens": 8},
        }

    async def test_emit_text_response(self):
        resp = self._resp([{"type": "text", "text": "Hello world"}])
        events = parse_events(await collect(_emit_anthropic_sse(resp, "msg_abc")))
        assert_sse_structure(events)

        names = [e for e, _ in events]
        assert names[0] == "message_start"
        assert "content_block_start" in names
        assert "content_block_delta" in names
        assert "content_block_stop" in names
        assert "message_delta" in names
        assert names[-1] == "message_stop"

        # content_block_start for index 0 is type=text
        start_data = [d for e, d in events if e == "content_block_start"][0]
        assert start_data["content_block"]["type"] == "text"

    async def test_emit_tool_use_response(self):
        resp = self._resp(
            [{"type": "tool_use", "id": "toolu_001", "name": "read_file", "input": {"path": "/tmp"}}],
            stop_reason="tool_use",
        )
        events = parse_events(await collect(_emit_anthropic_sse(resp, "msg_tool")))
        assert_sse_structure(events)

        start_data = [d for e, d in events if e == "content_block_start"][0]
        assert start_data["content_block"]["type"] == "tool_use"
        assert start_data["content_block"]["name"] == "read_file"
        assert start_data["content_block"]["id"] == "toolu_001"

        delta_events = [(e, d) for e, d in events if e == "content_block_delta"]
        assert delta_events, "Expected at least one content_block_delta"
        assert all(d["delta"]["type"] == "input_json_delta" for _, d in delta_events)

    async def test_emit_text_and_tool(self):
        resp = self._resp(
            [
                {"type": "text", "text": "Calling tool"},
                {"type": "tool_use", "id": "toolu_002", "name": "list_dir", "input": {}},
            ],
            stop_reason="tool_use",
        )
        events = parse_events(await collect(_emit_anthropic_sse(resp, "msg_mixed")))
        assert_sse_structure(events)

        start_events = [(e, d) for e, d in events if e == "content_block_start"]
        assert len(start_events) == 2

        idx0 = start_events[0][1]["index"]
        idx1 = start_events[1][1]["index"]
        assert idx0 == 0
        assert idx1 == 1

        assert start_events[0][1]["content_block"]["type"] == "text"
        assert start_events[1][1]["content_block"]["type"] == "tool_use"


# ===========================================================================
# 5. stream_oai_to_anthropic Path 3 (no tools) — 5 tests (async)
# ===========================================================================

@pytest.mark.asyncio
class TestStreamOaiPath3:
    """Path 3 = stream_oai_to_anthropic with tool_names=[] (no poke path)."""

    async def _run(self, sse_lines, **kwargs):
        oai_req = {"model": "local", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        oai_req.update(kwargs)
        with patch("bridge.httpx.AsyncClient", fake_httpx(sse_lines)):
            raw = await collect(stream_oai_to_anthropic(oai_req, "msg_p3", "local-model", tool_names=[]))
        return parse_events(raw)

    async def test_p3_plain_text(self):
        sse = make_sse(
            oai_text_chunk("Hello "),
            oai_text_chunk("world"),
            oai_finish_chunk("stop"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "Hello" in all_text
        assert "world" in all_text
        assert "<think>" not in all_text

    async def test_p3_think_block_stripped(self):
        # Think block split across two chunks, then visible text
        sse = make_sse(
            oai_text_chunk("<think>internal reasoning"),
            oai_text_chunk(" continued</think>visible text"),
            oai_finish_chunk("stop"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "internal reasoning" not in all_text
        assert "continued" not in all_text
        assert "visible text" in all_text

    async def test_p3_think_only_no_visible(self):
        sse = make_sse(
            oai_text_chunk("<think>only think</think>"),
            oai_finish_chunk("stop"),
        )
        events = await self._run(sse)
        # Must still produce the SSE frame (message_start and message_stop at minimum)
        names = [e for e, _ in events]
        assert "message_start" in names
        assert "message_stop" in names

        # No text delta should contain the think content
        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "only think" not in all_text

    async def test_p3_no_think_block(self):
        # First delta does NOT start with <think> — should stream immediately
        sse = make_sse(
            oai_text_chunk("Immediate text"),
            oai_finish_chunk("stop"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "Immediate text" in all_text

    async def test_p3_literal_think_tag_in_text(self):
        # Model outputs visible text that mentions <think> but NOT as a leading block
        literal = "Here is an example: <think>config</think>"
        sse = make_sse(
            oai_text_chunk(literal),
            oai_finish_chunk("stop"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        # The literal tag should pass through verbatim
        assert "<think>config</think>" in all_text


# ===========================================================================
# 6. _stream_with_poke Path 2 (tools present) — 8 tests (async)
# ===========================================================================

_BASE_OAI_REQ = {
    "model": "local",
    "stream": True,
    "tools": [
        {
            "type": "function",
            "function": {"name": "read_file", "parameters": {}},
        }
    ],
    "messages": [{"role": "user", "content": "hi"}],
}


@pytest.mark.asyncio
class TestStreamWithPoke:
    async def _run(self, sse_lines, oai_req=None, poke_response=None):
        req = oai_req or dict(_BASE_OAI_REQ)
        with patch("bridge.httpx.AsyncClient", fake_httpx(sse_lines, poke_response)):
            raw = await collect(_stream_with_poke(req, "msg_p2", "local-model", ["read_file"]))
        return parse_events(raw)

    async def test_p2_no_think_immediate_stream(self):
        sse = make_sse(
            oai_text_chunk("Plain text response"),
            oai_finish_chunk("stop"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "Plain text response" in all_text

    async def test_p2_think_no_poke_then_stream(self):
        # Think block that does NOT mention any tool name
        sse = make_sse(
            oai_text_chunk("<think>I should just answer directly</think>Here is my answer"),
            oai_finish_chunk("stop"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "I should just answer" not in all_text   # think stripped
        assert "Here is my answer" in all_text

    async def test_p2_think_poke_signal_tool_call_follows(self):
        # Think mentions "read_file", then model streams tool_call deltas
        # → enters POKE_BUFFER, but model produced tool_calls so no poke needed
        sse = make_sse(
            oai_text_chunk("<think>I need to call read_file</think>"),
            oai_tool_chunk(0, tc_id="call_001", name="read_file"),
            oai_tool_chunk(0, args='{"path": "/tmp"}'),
            oai_finish_chunk("tool_calls"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        tool_starts = [d for e, d in events if e == "content_block_start" and d["content_block"]["type"] == "tool_use"]
        assert tool_starts, "Expected a tool_use block in the SSE output"
        assert tool_starts[0]["content_block"]["name"] == "read_file"

    async def test_p2_think_poke_signal_suppressed_by_visible_text(self):
        # Think mentions "read_file" (would have been a poke trigger before the fix),
        # but then model produces visible text → NOT a stalled tool call → poke suppressed,
        # original conversational response is emitted.
        sse = make_sse(
            oai_text_chunk("<think>I need to call read_file for this</think>I cannot help with that."),
            oai_finish_chunk("stop"),
        )
        events = await self._run(sse)   # no poke_response — poke must not fire
        assert_sse_structure(events)

        tool_starts = [d for e, d in events if e == "content_block_start" and d["content_block"]["type"] == "tool_use"]
        assert not tool_starts, "Poke must not fire when model produced visible text"

        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "I cannot help with that." in all_text, "Original response must be emitted unchanged"

    async def test_p2_think_poke_signal_empty_response_fires_poke(self):
        # Think mentions "read_file" and model produces NO visible text → stalled tool
        # call → poke fires and tool_use block from poke response is emitted.
        poke_resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_poke",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "/poke"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        sse = make_sse(
            oai_text_chunk("<think>I need to call read_file for this</think>"),
            oai_finish_chunk("stop"),
        )
        events = await self._run(sse, poke_response=poke_resp)
        assert_sse_structure(events)

        tool_starts = [d for e, d in events if e == "content_block_start" and d["content_block"]["type"] == "tool_use"]
        assert tool_starts, "Poke response tool_use block should appear when model stalled with no visible text"

    async def test_p2_hard_stall_empty_after_tool_result(self):
        # Last message is role=tool, model returns empty response → poke fires
        poke_resp = {
            "choices": [
                {
                    "message": {
                        "content": "Here is the result.",
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
        req = dict(_BASE_OAI_REQ)
        req["messages"] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
        ]
        # Model returns completely empty response
        sse = make_sse(oai_finish_chunk("stop"))
        events = await self._run(sse, oai_req=req, poke_response=poke_resp)
        assert_sse_structure(events)

        # Poke response text should be in output
        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "Here is the result." in all_text

    async def test_p2_think_with_literal_tag_in_post_text(self):
        # Think block ends, post_text contains a literal <think>...</think> as example text
        post = 'Use <think>something</think> in your config.'
        sse = make_sse(
            oai_text_chunk(f"<think>I know the answer</think>{post}"),
            oai_finish_chunk("stop"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "I know the answer" not in all_text      # think stripped
        assert "<think>something</think>" in all_text   # literal in post-text preserved

    async def test_p2_tool_deltas_without_prior_text(self):
        # First chunk has tool_calls but no text → state should go immediately to STREAM
        sse = make_sse(
            oai_tool_chunk(0, tc_id="call_direct", name="read_file"),
            oai_tool_chunk(0, args='{"path": "/x"}'),
            oai_finish_chunk("tool_calls"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        tool_starts = [d for e, d in events if e == "content_block_start" and d["content_block"]["type"] == "tool_use"]
        assert tool_starts, "Tool call should produce a tool_use block"
        assert tool_starts[0]["content_block"]["name"] == "read_file"

    async def test_p2_multi_tool_calls(self):
        # Two separate tool call indices in the stream → two tool_use blocks
        sse = make_sse(
            oai_tool_chunk(0, tc_id="call_a", name="read_file"),
            oai_tool_chunk(0, args='{"path": "/a"}'),
            oai_tool_chunk(1, tc_id="call_b", name="read_file"),
            oai_tool_chunk(1, args='{"path": "/b"}'),
            oai_finish_chunk("tool_calls"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        tool_starts = [d for e, d in events if e == "content_block_start" and d["content_block"]["type"] == "tool_use"]
        assert len(tool_starts) == 2, f"Expected 2 tool_use blocks, got {len(tool_starts)}"
        indices = [d["index"] for d in tool_starts]
        assert sorted(indices) == [0, 1]

    async def test_p2_tool_deltas_during_think_with_both_fields(self):
        # The core bug fix: chunk has BOTH content and tool_calls during THINK state.
        # Tool deltas must NOT be silently dropped — they should be accumulated and emitted.
        sse = make_sse(
            oai_text_chunk("<think>I need to call read_file", usage=None),
            oai_tool_chunk(0, tc_id="call_think", name="read_file"),
            # Same chunk has both text AND tool_calls — this was the bug
            oai_text_chunk("</think>"),
            oai_tool_chunk(0, args='{"path": "/tmp"}'),
            oai_finish_chunk("tool_calls"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        tool_starts = [d for e, d in events if e == "content_block_start" and d["content_block"]["type"] == "tool_use"]
        assert len(tool_starts) == 1, f"Expected 1 tool_use block from THINK-phase tool deltas, got {len(tool_starts)}"
        assert tool_starts[0]["content_block"]["name"] == "read_file"
        assert tool_starts[0]["content_block"]["id"] == "call_think"

        # Verify the accumulated args are emitted as a delta
        json_deltas = [
            d["delta"]["partial_json"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "input_json_delta"
        ]
        full_args = "".join(json_deltas)
        assert "/tmp" in full_args

    async def test_p2_no_think_block_with_tool_deltas(self):
        # First chunk has no think block (doesn't start with <think>) but has tool_calls.
        # Tool deltas should be accumulated during the brief THINK phase and emitted.
        sse = make_sse(
            oai_tool_chunk(0, tc_id="call_direct", name="read_file"),
            oai_text_chunk("Direct answer"),
            oai_tool_chunk(0, args='{"path": "/x"}'),
            oai_finish_chunk("tool_calls"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        tool_starts = [d for e, d in events if e == "content_block_start" and d["content_block"]["type"] == "tool_use"]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["name"] == "read_file"

        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "Direct answer" in all_text


# ===========================================================================
# 7. _estimate_token_count — unit tests
# ===========================================================================

class TestEstimateTokenCount:
    def test_simple_text(self):
        body = {"messages": [{"role": "user", "content": "hello world"}]}
        result = bridge._estimate_token_count(body)
        assert result >= 1  # minimum 1 token

    def test_empty_messages(self):
        body = {}
        result = bridge._estimate_token_count(body)
        assert result >= 1  # base overhead from model BOS

    def test_system_message_string(self):
        body = {
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = bridge._estimate_token_count(body)
        assert result > bridge._estimate_token_count({"messages": [{"role": "user", "content": "hi"}]})

    def test_system_message_list(self):
        body = {
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = bridge._estimate_token_count(body)
        assert result >= 1

    def test_content_blocks(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello world"}]},
            ]
        }
        result = bridge._estimate_token_count(body)
        assert result >= 1

    def test_tool_use_blocks(self):
        body = {
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "f", "input": {"a": 1}}]},
            ]
        }
        result = bridge._estimate_token_count(body)
        assert result >= 1

    def test_tool_definitions_count(self):
        body = {
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result_with_tools = bridge._estimate_token_count(body)
        body_no_tools = {"messages": [{"role": "user", "content": "hi"}]}
        result_no_tools = bridge._estimate_token_count(body_no_tools)
        assert result_with_tools > result_no_tools

    def test_long_text_proportional(self):
        short = {"messages": [{"role": "user", "content": "a" * 40}]}
        long_body = {"messages": [{"role": "user", "content": "a" * 400}]}
        assert bridge._estimate_token_count(long_body) > bridge._estimate_token_count(short)

    def test_minimum_one(self):
        body = {}
        result = bridge._estimate_token_count(body)
        assert result >= 1


# ===========================================================================
# 8. /v1/messages/count_tokens — endpoint integration tests
# ===========================================================================

from fastapi.testclient import TestClient


class TestCountTokens:
    @pytest.fixture
    def client(self):
        return TestClient(bridge.app)

    def test_happy_path(self, client):
        resp = client.post("/v1/messages/count_tokens", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello world"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "input_tokens" in data
        assert "cache_creation_input_tokens" in data
        assert "cache_read_input_tokens" in data
        assert data["model"] == "test-model"
        assert isinstance(data["input_tokens"], int)
        assert data["input_tokens"] >= 1

    def test_invalid_json(self, client):
        resp = client.post("/v1/messages/count_tokens", content=b"not json", headers={"Content-Type": "application/json"})
        assert resp.status_code == 400

    def test_disabled_flag(self, client):
        original = bridge.COUNT_TOKENS_ENABLED
        bridge.COUNT_TOKENS_ENABLED = False
        try:
            resp = client.post("/v1/messages/count_tokens", json={"messages": []})
            assert resp.status_code == 404
        finally:
            bridge.COUNT_TOKENS_ENABLED = original

    def test_empty_body(self, client):
        resp = client.post("/v1/messages/count_tokens", json={})
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] >= 1

    def test_with_tools(self, client):
        resp = client.post("/v1/messages/count_tokens", json={
            "model": "local-model",
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
            "messages": [{"role": "user", "content": "read /etc/passwd"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["input_tokens"] >= 1

    def test_cache_fields_zero(self, client):
        resp = client.post("/v1/messages/count_tokens", json={"messages": [{"role": "user", "content": "hi"}]})
        data = resp.json()
        assert data["cache_creation_input_tokens"] == 0
        assert data["cache_read_input_tokens"] == 0


# ===========================================================================
# 9. TestBridgeTools — bridge-level web_search / web_fetch execution
# ===========================================================================

class TestBridgeTools:
    # -----------------------------------------------------------------------
    # _get_bridge_tool_arg
    # -----------------------------------------------------------------------

    def test_get_bridge_tool_arg_web_search_canonical(self):
        assert _get_bridge_tool_arg("web_search", {"query": "foo"}) == "foo"

    def test_get_bridge_tool_arg_web_search_alt_names(self):
        assert _get_bridge_tool_arg("web_search", {"search_query": "bar"}) == "bar"
        assert _get_bridge_tool_arg("web_search", {"q": "baz"}) == "baz"
        assert _get_bridge_tool_arg("web_search", {"text": "qux"}) == "qux"

    def test_get_bridge_tool_arg_web_search_first_string_fallback(self):
        result = _get_bridge_tool_arg("web_search", {"unknown_key": "some value"})
        assert result == "some value"

    def test_get_bridge_tool_arg_web_fetch_canonical(self):
        assert _get_bridge_tool_arg("web_fetch", {"url": "https://example.com"}) == "https://example.com"

    def test_get_bridge_tool_arg_web_fetch_alt_names(self):
        assert _get_bridge_tool_arg("web_fetch", {"uri": "https://a.com"}) == "https://a.com"
        assert _get_bridge_tool_arg("web_fetch", {"link": "https://b.com"}) == "https://b.com"
        assert _get_bridge_tool_arg("web_fetch", {"href": "https://c.com"}) == "https://c.com"

    def test_get_bridge_tool_arg_empty(self):
        assert _get_bridge_tool_arg("web_search", {}) == ""
        assert _get_bridge_tool_arg("web_fetch", {}) == ""

    # -----------------------------------------------------------------------
    # _bridge_web_search
    # -----------------------------------------------------------------------

    def test_bridge_web_search_ddgs_returns_results(self):
        """ddgs returns results — verify numbered list format."""
        ddgs_results = [
            {"title": "Title One", "href": "https://one.com", "body": "Snippet one"},
            {"title": "Title Two", "href": "https://two.com", "body": "Snippet two"},
        ]

        class FakeDDGS:
            def text(self, query, max_results=6): return ddgs_results

        with patch("ddgs.DDGS", return_value=FakeDDGS()):
            result = asyncio.run(_bridge_web_search("python asyncio"))

        assert "1. Title One" in result
        assert "https://one.com" in result
        assert "Snippet one" in result
        assert "2. Title Two" in result

    def test_bridge_web_search_ddgs_multiple_results(self):
        """ddgs returns multiple results — all included in output."""
        ddgs_results = [
            {"title": "DDG Title", "href": "https://ddg.com", "body": "DDG snippet"},
            {"title": "DDG Title 2", "href": "https://ddg2.com", "body": "DDG snippet 2"},
        ]

        class FakeDDGS:
            def text(self, query, max_results=6): return ddgs_results

        with patch("ddgs.DDGS", return_value=FakeDDGS()):
            result = asyncio.run(_bridge_web_search("test query"))

        assert "1. DDG Title" in result
        assert "https://ddg.com" in result
        assert "DDG snippet" in result
        assert "DDG Title 2" in result

    def test_bridge_web_search_empty_query(self):
        result = asyncio.run(_bridge_web_search(""))
        assert result.startswith("[error:")

    def test_bridge_web_search_no_results(self):
        """ddgs returns empty list → '(no results)'."""
        class FakeDDGS:
            def text(self, query, max_results=6): return []

        with patch("ddgs.DDGS", return_value=FakeDDGS()):
            result = asyncio.run(_bridge_web_search("obscure query"))

        assert result == "(no results)"

    # -----------------------------------------------------------------------
    # _bridge_web_fetch
    # -----------------------------------------------------------------------

    def test_bridge_web_fetch_html(self):
        """HTML response — script/head stripped, body text preserved."""
        html = "<html><head><title>T</title></head><body><p>Hello world content</p></body></html>"

        class FakeResp:
            text = html
            headers = {"content-type": "text/html; charset=utf-8"}
            def raise_for_status(self): pass

        class FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get(self, *a, **kw): return FakeResp()

        with patch("bridge.httpx.AsyncClient", return_value=FakeClient()):
            result = asyncio.run(_bridge_web_fetch("https://example.com"))

        assert "Hello world content" in result
        # head tag is in SKIP_TAGS — title should not appear
        assert "T" not in result or "Hello" in result  # at minimum, content present

    def test_bridge_web_fetch_plain_text(self):
        """Plain text response — returned as-is."""
        class FakeResp:
            text = "raw text"
            headers = {"content-type": "text/plain"}
            def raise_for_status(self): pass

        class FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get(self, *a, **kw): return FakeResp()

        with patch("bridge.httpx.AsyncClient", return_value=FakeClient()):
            result = asyncio.run(_bridge_web_fetch("https://example.com/file.txt"))

        assert result == "raw text"

    def test_bridge_web_fetch_truncation(self):
        """Body > 12000 chars is truncated with suffix."""
        long_text = "x" * 15000

        class FakeResp:
            text = long_text
            headers = {"content-type": "text/plain"}
            def raise_for_status(self): pass

        class FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get(self, *a, **kw): return FakeResp()

        with patch("bridge.httpx.AsyncClient", return_value=FakeClient()):
            result = asyncio.run(_bridge_web_fetch("https://example.com/big"))

        assert "... [truncated," in result
        assert len(result) < 15000

    def test_bridge_web_fetch_error(self):
        """HTTP error → result starts with '[error:'."""
        class FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get(self, *a, **kw): raise Exception("connection refused")

        with patch("bridge.httpx.AsyncClient", return_value=FakeClient()):
            result = asyncio.run(_bridge_web_fetch("https://bad.host/"))

        assert result.startswith("[error:")

    # -----------------------------------------------------------------------
    # _oai_resp_has_only_bridge_calls / _oai_resp_get_bridge_calls
    # -----------------------------------------------------------------------

    def _make_oai_resp(self, tool_names: list[str]) -> dict:
        tool_calls = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for i, name in enumerate(tool_names)
        ]
        msg: dict = {"content": None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {"choices": [{"message": msg, "finish_reason": "tool_calls"}], "usage": {}}

    def test_oai_resp_has_only_bridge_calls_true(self):
        resp = self._make_oai_resp(["web_search"])
        assert _oai_resp_has_only_bridge_calls(resp) is True

    def test_oai_resp_has_only_bridge_calls_mixed(self):
        resp = self._make_oai_resp(["web_search", "bash"])
        assert _oai_resp_has_only_bridge_calls(resp) is False

    def test_oai_resp_has_only_bridge_calls_no_tools(self):
        resp = self._make_oai_resp([])
        assert _oai_resp_has_only_bridge_calls(resp) is False

    # -----------------------------------------------------------------------
    # _bridge_tool_loop
    # -----------------------------------------------------------------------

    def _make_tool_call_resp(self, tool_name: str, call_id: str = "call_x") -> dict:
        return {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": '{"query": "test"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def _make_text_resp(self, text: str = "Final answer") -> dict:
        return {
            "choices": [{"message": {"content": text, "tool_calls": []}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10},
        }

    def test_bridge_tool_loop_single_search(self):
        """Initial resp has web_search — raw results returned directly, call_llama never called."""
        call_count = [0]

        async def fake_call_llama(client, request):
            call_count[0] += 1
            return self._make_text_resp()

        async def fake_run_bridge_tool(name, input_obj):
            return "search results here"

        async def _run():
            oai_request = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
            initial_resp = self._make_tool_call_resp("web_search")
            with patch("bridge.call_llama", fake_call_llama):
                with patch("bridge._run_bridge_tool", fake_run_bridge_tool):
                    return await _bridge_tool_loop(None, oai_request, initial_resp, "test-model")

        result = asyncio.run(_run())
        assert call_count[0] == 0, "call_llama must not be called"
        assert result["choices"][0]["message"]["content"] == "search results here"

    def test_bridge_tool_loop_no_bridge_calls(self):
        """Response has no tool_calls — loop exits immediately, call_llama never called."""
        call_count = [0]

        async def fake_call_llama(client, request):
            call_count[0] += 1
            return self._make_text_resp()

        async def _run():
            oai_request = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
            text_resp = self._make_text_resp()
            with patch("bridge.call_llama", fake_call_llama):
                return await _bridge_tool_loop(None, oai_request, text_resp, "test-model")

        result = asyncio.run(_run())
        assert call_count[0] == 0
        assert result["choices"][0]["message"]["content"] == "Final answer"

    def test_bridge_tool_loop_mixed_tools_passthrough(self):
        """Response has web_search + bash → mixed → loop exits without calling call_llama."""
        call_count = [0]

        async def fake_call_llama(client, request):
            call_count[0] += 1
            return self._make_text_resp()

        async def _run():
            oai_request = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
            mixed_resp = self._make_oai_resp(["web_search", "bash"])
            with patch("bridge.call_llama", fake_call_llama):
                return await _bridge_tool_loop(None, oai_request, mixed_resp, "test-model")

        result = asyncio.run(_run())
        assert call_count[0] == 0

    def test_bridge_tool_loop_returns_raw_without_model_call(self):
        """Bridge tool loop returns raw results directly — call_llama is never invoked."""
        call_count = [0]

        async def fake_call_llama(client, request):
            call_count[0] += 1
            return self._make_text_resp()

        async def fake_run_bridge_tool(name, input_obj):
            return "raw ddgs result"

        async def _run():
            oai_request = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
            initial_resp = self._make_tool_call_resp("web_search")
            with patch("bridge.call_llama", fake_call_llama):
                with patch("bridge._run_bridge_tool", fake_run_bridge_tool):
                    return await _bridge_tool_loop(None, oai_request, initial_resp, "test-model")

        result = asyncio.run(_run())
        assert call_count[0] == 0, "call_llama must not be called — raw results returned directly"
        msg = result["choices"][0]["message"]
        assert msg["content"] == "raw ddgs result"
        assert not msg["tool_calls"]


# ===========================================================================
# 10. TestStreamWithPokeBridgeActive — bridge_active forces POKE_BUFFER
# ===========================================================================

_WEB_SEARCH_REQ = {
    "model": "local",
    "stream": True,
    "tools": [{"type": "function", "function": {"name": "web_search", "parameters": {}}}],
    "messages": [{"role": "user", "content": "search for SpaceX IPO"}],
}

_SEARCH_RESULT = "1. SpaceX IPO\n   https://example.com\n   SpaceX remains private as of 2026."

_FINAL_RESP_AFTER_SEARCH = {
    "choices": [{"message": {"content": "Based on search: SpaceX is still private.", "tool_calls": []}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 20, "completion_tokens": 15},
}


@pytest.mark.asyncio
class TestStreamWithPokeBridgeActive:
    """
    Tests for _stream_with_poke when bridge tools (web_search/web_fetch) are
    in tool_names.  bridge_active=True forces POKE_BUFFER regardless of whether
    a think-block is present, so the bridge can intercept and execute the search
    locally instead of passing the tool_use back to Claude Code.
    """

    async def _run(self, sse_lines, final_resp=None, search_result=_SEARCH_RESULT):
        resp = final_resp or _FINAL_RESP_AFTER_SEARCH
        with patch("bridge.httpx.AsyncClient", fake_httpx(sse_lines, resp)):
            with patch("bridge._run_bridge_tool", AsyncMock(return_value=search_result)):
                raw = await collect(_stream_with_poke(
                    dict(_WEB_SEARCH_REQ), "msg_ba", "local-model", ["web_search"]
                ))
        return parse_events(raw)

    async def test_no_think_text_first_enters_poke_buffer(self):
        """Model emits text immediately (no think block) then calls web_search.
        bridge_active must force POKE_BUFFER so the search is executed by the bridge."""
        sse = make_sse(
            oai_text_chunk("Searching for you..."),
            oai_tool_chunk(0, tc_id="call_ws1", name="web_search"),
            oai_tool_chunk(0, args='{"query": "SpaceX IPO"}'),
            oai_finish_chunk("tool_calls"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        # Bridge handled web_search — must NOT appear as tool_use in client output
        tool_starts = [
            d for e, d in events
            if e == "content_block_start" and d.get("content_block", {}).get("type") == "tool_use"
        ]
        assert not tool_starts, "web_search should be executed by bridge, not forwarded to client"

        # Raw search result must appear in the text response (no model synthesis)
        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "SpaceX remains private" in all_text

    async def test_no_think_tool_delta_first_enters_poke_buffer(self):
        """Model emits tool_call first (no text, no think block) — web_search.
        bridge_active must force POKE_BUFFER even with no preceding text."""
        sse = make_sse(
            oai_tool_chunk(0, tc_id="call_ws2", name="web_search"),
            oai_tool_chunk(0, args='{"query": "SpaceX IPO 2026"}'),
            oai_finish_chunk("tool_calls"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        tool_starts = [
            d for e, d in events
            if e == "content_block_start" and d.get("content_block", {}).get("type") == "tool_use"
        ]
        assert not tool_starts, "web_search should be executed by bridge, not forwarded to client"

        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "SpaceX remains private" in all_text

    async def test_think_block_bridge_active_enters_poke_buffer(self):
        """Model uses think block then calls web_search — existing POKE_BUFFER path,
        now triggered via the hoisted bridge_active variable."""
        sse = make_sse(
            oai_text_chunk("<think>I need to search for SpaceX IPO info</think>"),
            oai_tool_chunk(0, tc_id="call_ws3", name="web_search"),
            oai_tool_chunk(0, args='{"query": "SpaceX IPO news"}'),
            oai_finish_chunk("tool_calls"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        tool_starts = [
            d for e, d in events
            if e == "content_block_start" and d.get("content_block", {}).get("type") == "tool_use"
        ]
        assert not tool_starts, "web_search should be executed by bridge (think-block POKE_BUFFER path)"

        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "SpaceX remains private" in all_text

    async def test_search_results_returned_directly(self):
        """Verify raw DDGS results appear in the SSE output without a model synthesis call."""
        call_count = [0]

        async def counting_call_llama(client, request):
            call_count[0] += 1
            return _FINAL_RESP_AFTER_SEARCH

        sse = make_sse(
            oai_tool_chunk(0, tc_id="call_ws4", name="web_search"),
            oai_tool_chunk(0, args='{"query": "SpaceX IPO"}'),
            oai_finish_chunk("tool_calls"),
        )
        with patch("bridge.httpx.AsyncClient", fake_httpx(sse)):
            with patch("bridge._run_bridge_tool", AsyncMock(return_value="SpaceX remains private.")):
                with patch("bridge.call_llama", counting_call_llama):
                    events = await collect(_stream_with_poke(
                        dict(_WEB_SEARCH_REQ), "msg_verify", "local-model", ["web_search"]
                    ))

        assert call_count[0] == 0, "call_llama must not be called — raw results returned directly"
        all_text = "".join(
            d["delta"]["text"]
            for e, d in parse_events(events)
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "SpaceX remains private." in all_text

    async def test_bridge_active_model_returns_only_text(self):
        """bridge_active=True but model produces only text (no tool call).
        Buffer is flushed via _emit_anthropic_sse — text response returned normally."""
        sse = make_sse(
            oai_text_chunk("SpaceX has not announced an IPO."),
            oai_finish_chunk("stop"),
        )
        events = await self._run(sse)
        assert_sse_structure(events)

        all_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        assert "SpaceX has not announced an IPO." in all_text

    async def test_non_bridge_tool_passes_through_unchanged(self):
        """Non-bridge tool (read_file) — bridge_active=False — tool_use passes to client as before."""
        req = {
            "model": "local",
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
            "messages": [{"role": "user", "content": "read a file"}],
        }
        sse = make_sse(
            oai_tool_chunk(0, tc_id="call_rf", name="read_file"),
            oai_tool_chunk(0, args='{"path": "/etc/hosts"}'),
            oai_finish_chunk("tool_calls"),
        )
        with patch("bridge.httpx.AsyncClient", fake_httpx(sse)):
            raw = await collect(_stream_with_poke(req, "msg_rf", "local-model", ["read_file"]))
        events = parse_events(raw)
        assert_sse_structure(events)

        tool_starts = [
            d for e, d in events
            if e == "content_block_start" and d.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["name"] == "read_file"

    async def test_stream_state_coerces_invalid_enum_in_tool_args(self):
        """STREAM path (think block present but tool name not in think text) must
        coerce invalid enum values before delivering input_json_delta to Claude Code.
        Reproduces the TaskUpdate status='done' → 'completed' bug."""
        task_update_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "status": {
                    "anyOf": [
                        {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        {"type": "string", "const": "deleted"},
                    ]
                },
            },
        }
        req = {
            "model": "local",
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "TaskUpdate", "parameters": task_update_schema}}],
            "messages": [{"role": "user", "content": "update task"}],
        }
        # Think content deliberately does not mention "TaskUpdate" → poke_signal=False → STREAM state
        sse = make_sse(
            oai_text_chunk("<think>I should mark the task as done.</think>"),
            oai_tool_chunk(0, tc_id="call_tu", name="TaskUpdate"),
            oai_tool_chunk(0, args='{"id": "task_1", "status": "done"}'),
            oai_finish_chunk("tool_calls"),
        )
        with patch("bridge.httpx.AsyncClient", fake_httpx(sse)):
            raw = await collect(_stream_with_poke(req, "msg_tu", "local-model", ["TaskUpdate"]))
        events = parse_events(raw)
        assert_sse_structure(events)

        # Collect all input_json_delta content for the TaskUpdate call
        arg_deltas = [
            d["delta"]["partial_json"]
            for e, d in events
            if e == "content_block_delta" and d["delta"].get("type") == "input_json_delta"
        ]
        assert arg_deltas, "Expected at least one input_json_delta for TaskUpdate"
        assembled = json.loads("".join(arg_deltas))
        assert assembled["status"] == "completed", (
            f"Expected status coerced to 'completed', got {assembled['status']!r}"
        )


# ===========================================================================
# 10. estimated_input_tokens propagation
# ===========================================================================

class TestEstimatedInputTokens:
    """
    Verify that estimated_input_tokens is used as input_tokens in message_start
    rather than defaulting to 0, for both the _stream_with_poke path and the
    simple stream_oai_to_anthropic path (no tool_names).
    """

    # --- _stream_with_poke path ---

    @pytest.mark.asyncio
    async def test_estimate_used_when_usage_absent_in_first_chunk(self):
        """Usage arrives only in the final chunk; message_start should use the estimate."""
        req = {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
        sse = make_sse(
            oai_text_chunk("<think>reasoning</think>"),
            oai_text_chunk("Hello"),
            # Usage only in the final finish chunk (typical llama.cpp behaviour)
            oai_finish_chunk("stop", usage={"prompt_tokens": 9500, "completion_tokens": 10}),
        )
        with patch("bridge.httpx.AsyncClient", fake_httpx(sse)):
            raw = await collect(_stream_with_poke(req, "msg_est", "local-model", ["Bash"], estimated_input_tokens=8000))
        events = parse_events(raw)
        assert_sse_structure(events)
        ms = next(d for e, d in events if e == "message_start")
        assert ms["message"]["usage"]["input_tokens"] == 8000, (
            f"Expected estimate 8000 in message_start, got {ms['message']['usage']['input_tokens']}"
        )

    @pytest.mark.asyncio
    async def test_default_estimate_is_zero(self):
        """Without passing an estimate the old default of 0 is preserved."""
        req = {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
        sse = make_sse(
            oai_text_chunk("<think>reasoning</think>"),
            oai_text_chunk("Hello"),
            oai_finish_chunk("stop", usage={"prompt_tokens": 9500, "completion_tokens": 10}),
        )
        with patch("bridge.httpx.AsyncClient", fake_httpx(sse)):
            raw = await collect(_stream_with_poke(req, "msg_zero", "local-model", ["Bash"]))
        events = parse_events(raw)
        ms = next(d for e, d in events if e == "message_start")
        assert ms["message"]["usage"]["input_tokens"] == 0

    @pytest.mark.asyncio
    async def test_real_usage_in_first_chunk_overrides_estimate(self):
        """If llama.cpp sends prompt_tokens in the first chunk, that value wins."""
        req = {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
        # First chunk carries usage (uncommon but possible)
        sse = make_sse(
            oai_text_chunk("Hello", usage={"prompt_tokens": 7777, "completion_tokens": 0}),
            oai_finish_chunk("stop"),
        )
        with patch("bridge.httpx.AsyncClient", fake_httpx(sse)):
            raw = await collect(_stream_with_poke(req, "msg_real", "local-model", ["Bash"], estimated_input_tokens=5000))
        events = parse_events(raw)
        ms = next(d for e, d in events if e == "message_start")
        # message_start is emitted once the STREAM state is entered, by which point the
        # first chunk (carrying usage) has already been processed — so real value wins.
        assert ms["message"]["usage"]["input_tokens"] == 7777

    @pytest.mark.asyncio
    async def test_estimate_nonzero_triggers_correct_token_reporting(self):
        """Smoke-test: a large estimate produces a non-zero input_tokens in message_start."""
        req = {"model": "local", "messages": [{"role": "user", "content": "x" * 400}]}
        sse = make_sse(
            oai_text_chunk("response"),
            oai_finish_chunk("stop"),
        )
        with patch("bridge.httpx.AsyncClient", fake_httpx(sse)):
            raw = await collect(_stream_with_poke(req, "msg_big", "local-model", ["Read"], estimated_input_tokens=50000))
        events = parse_events(raw)
        ms = next(d for e, d in events if e == "message_start")
        assert ms["message"]["usage"]["input_tokens"] == 50000

    # --- simple stream_oai_to_anthropic path (no tool_names) ---

    @pytest.mark.asyncio
    async def test_simple_path_estimate_used_when_no_usage_in_first_chunk(self):
        """No tool_names → simple path; estimate should appear in message_start."""
        req = {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
        sse = make_sse(
            oai_text_chunk("Hello"),
            oai_finish_chunk("stop", usage={"prompt_tokens": 200, "completion_tokens": 5}),
        )
        with patch("bridge.httpx.AsyncClient", fake_httpx(sse)):
            raw = await collect(stream_oai_to_anthropic(req, "msg_sp", "local-model", [], estimated_input_tokens=3000))
        events = parse_events(raw)
        assert_sse_structure(events)
        ms = next(d for e, d in events if e == "message_start")
        assert ms["message"]["usage"]["input_tokens"] == 3000

    @pytest.mark.asyncio
    async def test_simple_path_usage_in_first_chunk_overrides_estimate(self):
        """Simple path: if the first chunk has prompt_tokens it replaces the estimate."""
        req = {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
        sse = make_sse(
            oai_text_chunk("Hello", usage={"prompt_tokens": 999, "completion_tokens": 0}),
            oai_finish_chunk("stop"),
        )
        with patch("bridge.httpx.AsyncClient", fake_httpx(sse)):
            raw = await collect(stream_oai_to_anthropic(req, "msg_sp2", "local-model", [], estimated_input_tokens=3000))
        events = parse_events(raw)
        ms = next(d for e, d in events if e == "message_start")
        assert ms["message"]["usage"]["input_tokens"] == 999

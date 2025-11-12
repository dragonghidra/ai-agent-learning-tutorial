# pip install -U langchain langchain-openai requests pydantic
# export OPENAI_API_KEY="sk-..."   # or set in your environment

import argparse
import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal

import requests

import cli_ui
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage

# --------- Real tools (they perform actual HTTP requests) ----------

HTTP_MAX_RESPONSE_CHARS = int(os.environ.get("HTTP_TOOL_MAX_CHARS", "120000"))
HTTP_USER_AGENT = os.environ.get(
    "HTTP_TOOL_USER_AGENT", "LangGraphWeatherAgent/1.0 (+https://github.com/bo)"
)
_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_EXTRAS = {
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/javascript",
}


def _looks_textual(content_type: str | None) -> bool:
    if not content_type:
        return True
    mime = content_type.split(";", 1)[0].strip().lower()
    return mime in _TEXT_MIME_EXTRAS or any(mime.startswith(p) for p in _TEXT_MIME_PREFIXES)


@tool
def http_get(url: str) -> str:
    """Fetch a URL and return (at most) HTTP_MAX_RESPONSE_CHARS of text."""
    headers = {"User-Agent": HTTP_USER_AGENT}
    with requests.get(url, timeout=15, headers=headers, stream=True) as resp:
        resp.raise_for_status()
        if not _looks_textual(resp.headers.get("Content-Type")):
            ctype = resp.headers.get("Content-Type", "unknown")
            return f"Unsupported content type '{ctype}'. Only text responses are returned."

        resp.encoding = resp.encoding or resp.apparent_encoding or "utf-8"
        remaining = HTTP_MAX_RESPONSE_CHARS
        chunks: list[str] = []
        for chunk in resp.iter_content(chunk_size=8192, decode_unicode=True):
            if not chunk:
                continue
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                remaining = 0
                break
            chunks.append(chunk)
            remaining -= len(chunk)
            if remaining <= 0:
                break

    body = "".join(chunks).strip()
    if not body:
        return "[Empty response]"

    if remaining <= 0:
        body += f"\n\n[Truncated to first {HTTP_MAX_RESPONSE_CHARS:,} characters]"
    return body


@tool
def get_weather(location: str, units: Literal["us", "metric"] = "us") -> str:
    """Get current weather for a place name using the Open-Meteo API.
    Args:
        location: A city or place name (e.g., "San Francisco", "Paris, FR").
        units: "us" (°F, mph) or "metric" (°C, km/h).
    Returns:
        A short human-readable summary of current conditions.
    """
    geo_resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    geo_resp.raise_for_status()
    geo = geo_resp.json()
    if not geo.get("results"):
        return f"Couldn't find '{location}'. Try a more specific name."

    place = geo["results"][0]
    lat, lon = place["latitude"], place["longitude"]
    resolved = ", ".join(
        p for p in [place.get("name"), place.get("admin1"), place.get("country")] if p
    )

    temp_unit = "fahrenheit" if units == "us" else "celsius"
    wind_unit = "mph" if units == "us" else "kmh"

    current_vars = ",".join(
        [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "precipitation",
            "cloud_cover",
        ]
    )

    wx_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": current_vars,
            "temperature_unit": temp_unit,
            "wind_speed_unit": wind_unit,
            "timezone": "auto",
        },
        timeout=10,
    )
    wx_resp.raise_for_status()
    wx = wx_resp.json().get("current", {})

    def get(key, default="—"):
        return wx.get(key, default)

    WMO = {
        0: "Clear",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Drizzle",
        55: "Heavy drizzle",
        61: "Light rain",
        63: "Rain",
        65: "Heavy rain",
        71: "Light snow",
        73: "Snow",
        75: "Heavy snow",
        80: "Rain showers",
        81: "Heavy rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm w/ hail",
        99: "Severe thunderstorm w/ hail",
    }

    code = int(get("weather_code", -1)) if str(get("weather_code", "")).isdigit() else -1
    desc = WMO.get(code, "Unknown")

    return (
        f"{resolved} — {get('temperature_2m')}°{'F' if units=='us' else 'C'}, "
        f"RH {get('relative_humidity_2m')}%, "
        f"wind {get('wind_speed_10m')} {wind_unit} "
        f"({get('wind_direction_10m')}°), "
        f"precip {get('precipitation')} mm, cloud {get('cloud_cover')}%, "
        f"{desc}. "
        f"[{get('time','now')}]"
    )


TOOLS = [http_get, get_weather]

# --------- Real LLM (tool-calling enabled) ----------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment before running this script.")

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    api_key=OPENAI_API_KEY,
)

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt="You are a helpful assistant.",
)

# --------- Runtime helpers & interactive entrypoints ---------

CLI_READY = threading.Event()
API_HOST = os.environ.get("WEATHER_AGENT_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("WEATHER_AGENT_API_PORT", "8080"))


def env_flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def make_json_safe(value: Any):
    if isinstance(value, BaseMessage):
        payload = {
            "type": value.type,
            "content": value.content,
        }
        if getattr(value, "name", None):
            payload["name"] = value.name
        if getattr(value, "tool_calls", None):
            payload["tool_calls"] = value.tool_calls
        return make_json_safe(payload)
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def ensure_message_content(value: Any):
    """Return content that satisfies the str-or-list constraint for LangChain messages."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, (str, dict)) for item in value):
        return value
    try:
        return json.dumps(make_json_safe(value), ensure_ascii=False)
    except TypeError:
        return str(value)


SSE_FORCE = env_flag("WEATHER_AGENT_FORCE_SSE")
IDE_INTEGRATION = env_flag("ENABLE_IDE_INTEGRATION")
SSE_PORT_PRESENT = bool(os.environ.get("CLAUDE_CODE_SSE_PORT"))
INTERACTIVE_SHELL_ENABLED = SSE_FORCE or IDE_INTEGRATION or SSE_PORT_PRESENT
VERBOSE = False


class InteractiveShellStreamer:
    """Minimal SSE emitter for the Codex interactive shell."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._write_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._next_session_id = 0

    def begin(self, *, source: str, prompt: str, metadata: dict | None = None):
        if not self.enabled:
            return None
        with self._session_lock:
            self._next_session_id += 1
            session_id = self._next_session_id
        payload = {"session_id": session_id, "source": source, "prompt": prompt}
        if metadata:
            payload.update(metadata)
        self._emit("response_start", payload)
        return session_id

    def message(self, message, session_id: int | None = None):
        if not self.enabled:
            return
        payload = serialize_message(message)
        if session_id is not None:
            payload["session_id"] = session_id
        self._emit("response_message", payload)

    def end(self, session_id: int | None = None, status: str = "ok", metadata: dict | None = None):
        if not self.enabled:
            return
        payload = {"status": status}
        if session_id is not None:
            payload["session_id"] = session_id
        if metadata:
            payload.update(metadata)
        self._emit("response_end", payload)

    def error(self, session_id: int | None, message: str):
        if not self.enabled:
            return
        payload = {"error": message}
        if session_id is not None:
            payload["session_id"] = session_id
        self._emit("response_error", payload)

    def _emit(self, event: str, payload):
        safe_payload = make_json_safe(payload)
        with self._write_lock:
            print(f"event: {event}", flush=True)
            print(f"data: {json.dumps(safe_payload, ensure_ascii=False)}", flush=True)
            print(flush=True)


SHELL_STREAMER = InteractiveShellStreamer(INTERACTIVE_SHELL_ENABLED)


class ConversationManager:
    """Thread-safe wrapper that keeps conversation state in sync with the agent."""

    def __init__(self, agent_executor):
        self.agent = agent_executor
        self.history: list[BaseMessage] = []
        self._lock = threading.Lock()

    @contextmanager
    def locked_submit(self, content: str, source: str = "cli"):
        """Run the agent while holding the manager lock until responses are consumed."""

        with self._lock:
            self.history.append(HumanMessage(content=ensure_message_content(content), name=source))
            reply_start = len(self.history)
            result = self.agent.invoke({"messages": self.history})
            messages = result.get("messages") if isinstance(result, dict) else None
            if messages:
                self.history = list(messages)
            responses = [msg for msg in self.history[reply_start:] if msg.type != "human"]
            yield responses

    def submit(self, content: str, source: str = "cli"):
        """Compatibility helper for callers that don't need extended locking."""

        with self.locked_submit(content, source) as responses:
            return responses


def stringify_content(content):
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except TypeError:
        return str(content)


def _message_style(message_type: str) -> str:
    return {
        "ai": "assistant",
        "assistant": "assistant",
        "tool": "tool",
        "system": "muted",
        "human": "user",
    }.get(message_type, "accent")


def print_message(message):
    """Pretty-print one message plus any tool calls."""
    label = message.type.upper()
    if getattr(message, "name", None):
        label = f"{label}[{message.name}]"
    label_text = cli_ui.color_text(label.ljust(12), style=_message_style(message.type), bold=True)
    content = stringify_content(getattr(message, "content", ""))
    print(f"{label_text} {content}")
    for tc in getattr(message, "tool_calls", []) or []:
        tool_label = cli_ui.color_text("  -> tool_call", style="tool")
        print(f"{tool_label} {stringify_content(tc)}")


def _extract_final_ai(messages):
    final_ai = None
    for msg in messages:
        if getattr(msg, "type", None) == "ai":
            final_ai = msg
    return final_ai


def _print_pretty_ai(message):
    if not message:
        return
    content = stringify_content(getattr(message, "content", "")).strip()
    if not content:
        return
    cli_ui.print_panel("Assistant", content, style="assistant")


def display_responses(messages, *, streamer=None, session_id=None, verbose=None):
    is_verbose = VERBOSE if verbose is None else verbose

    if is_verbose:
        for msg in messages:
            print_message(msg)
            if streamer is not None:
                streamer.message(msg, session_id=session_id)
        return

    final_ai = _extract_final_ai(messages)
    if final_ai:
        _print_pretty_ai(final_ai)
        if streamer is not None:
            streamer.message(final_ai, session_id=session_id)
    else:
        # Fall back to verbose output when no assistant reply is present.
        for msg in messages:
            print_message(msg)
            if streamer is not None:
                streamer.message(msg, session_id=session_id)


def reprint_prompt():
    if CLI_READY.is_set():
        print(cli_ui.prompt_label("You"), end="", flush=True)


def run_cli_chat(conversation: ConversationManager, stop_event: threading.Event):
    """Interactive multi-turn chat loop in the terminal."""
    cli_ui.print_banner(
        "LangChain Weather",
        "LangChain agent with real tool calls and optional verbose traces.",
    )
    cli_ui.print_status("Type 'exit' or 'quit' to leave. Use --verbose for tool traces.", kind="info")
    CLI_READY.set()

    try:
        while not stop_event.is_set():
            try:
                user_text = input(cli_ui.prompt_label("You")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                cli_ui.print_status("Exiting.", kind="warning")
                stop_event.set()
                break

            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                cli_ui.print_status("Goodbye!", kind="success")
                stop_event.set()
                break

            session_id = SHELL_STREAMER.begin(
                source="cli",
                prompt=user_text,
                metadata={"origin": "cli"}
            ) if SHELL_STREAMER.enabled else None

            try:
                with conversation.locked_submit(user_text, source="cli") as responses:
                    display_responses(
                        responses,
                        streamer=SHELL_STREAMER if SHELL_STREAMER.enabled else None,
                        session_id=session_id,
                    )
            except Exception as exc:  # noqa: BLE001
                cli_ui.print_status(f"[cli] agent error: {exc}", kind="error")
                if session_id is not None:
                    SHELL_STREAMER.error(session_id, str(exc))
            else:
                if session_id is not None:
                    SHELL_STREAMER.end(session_id=session_id, metadata={"origin": "cli"})
    finally:
        CLI_READY.clear()


def serialize_message(message):
    payload = {
        "type": message.type,
        "name": getattr(message, "name", None),
    }
    content = getattr(message, "content", "")
    payload["content"] = make_json_safe(content)
    if getattr(message, "tool_calls", None):
        payload["tool_calls"] = make_json_safe(message.tool_calls)
    return payload


def start_background_api(conversation: ConversationManager, host: str, port: int):
    """Launch a simple HTTP endpoint so external processes can inject prompts."""

    conv = conversation

    class ConversationHandler(BaseHTTPRequestHandler):
        conversation = conv

        def do_POST(self):
            if self.path != "/chat":
                self.send_error(404, "POST /chat to talk to the agent.")
                return

            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON payload.")
                return

            message = payload.get("message")
            if not message:
                self.send_error(400, "Field 'message' is required.")
                return

            source = str(payload.get("source", "api"))
            try:
                responses = self.conversation.submit(str(message), source=source)
            except Exception as exc:  # noqa: BLE001
                self.send_error(500, f"Agent error: {exc}")
                return

            response_body = json.dumps(
                {"responses": [serialize_message(msg) for msg in responses]},
                ensure_ascii=True,
            ).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, fmt, *args):  # noqa: D401
            print(f"[api] {fmt % args}")

    server = ThreadingHTTPServer((host, port), ConversationHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cli_ui.print_status(
        f"REST API listening on http://{host}:{port}/chat (POST {{'message': '...'}})",
        kind="info",
    )
    return server



def parse_args():
    parser = argparse.ArgumentParser(description="LangChain weather agent CLI.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every intermediate message and tool call.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    conversation = ConversationManager(agent)
    stop_event = threading.Event()

    api_server = None
    try:
        api_server = start_background_api(conversation, API_HOST, API_PORT)
    except OSError as exc:
        cli_ui.print_status(
            f"[api] Unable to start server on {API_HOST}:{API_PORT}: {exc}",
            kind="warning",
        )

    try:
        run_cli_chat(conversation, stop_event)
    finally:
        stop_event.set()
        if api_server:
            api_server.shutdown()
            api_server.server_close()


if __name__ == "__main__":
    main()

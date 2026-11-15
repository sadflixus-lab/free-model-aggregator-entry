#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free-tier providers.

The server reads the trial brief (MAKEFUN_PROMPT_FILE): a provider directory,
a quality table and a workload manifest. It probes every provider's catalog
route in that provider's own wire, registers one catalog with the env, then
answers each client request with the highest-rated free model that meets the
request's requirements and still answers -- failing over to the next best
candidate whenever a provider refuses (402/429/5xx).

Design notes:
  * Upstream calls are always non-streaming; the answer is re-rendered in the
    client's wire and format (JSON or SSE). One internal answer shape keeps
    the four renderers small and exact.
  * Cross-wire translation is structural: system text, message parts, tool
    names, max_tokens, stop sequences and JSON mode are carried field by
    field, because the providers echo what they parsed and the client
    compares that echo with what it sent.
  * Everything is a function of the brief, the requests and the providers'
    answers: no wall clock (fixed timestamps), no randomness.
  * Python standard library only.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

PROTOCOL_VERSION = 1
PROBE_TIMEOUT_S = 5.0
CALL_TIMEOUT_S = 20.0
POLL_INTERVAL_S = 0.1
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return isinstance(value, dict)


def string_of(value):
    return value if isinstance(value, str) else ""


def number_or_none(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value == value and value not in (float("inf"), float("-inf")):
        return value
    return None


# --------------------------------------------------------------------------
# The brief: three JSON documents found by their `schema` member.
# --------------------------------------------------------------------------

def documents_in(text):
    """Every JSON object in `text` carrying a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def object_end(text, start):
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


class Provider:
    __slots__ = ("id", "name", "wire", "auth", "base", "key_env", "key")

    def __init__(self, raw):
        self.id = string_of(raw.get("id"))
        self.name = string_of(raw.get("name")) or self.id
        self.wire = raw.get("wire")
        self.auth = raw.get("auth") if raw.get("auth") in (
            "bearer", "x-api-key", "x-goog-api-key", "none") else "bearer"
        base = ""
        for field in ("path", "base_path", "base_url", "base"):
            if string_of(raw.get(field)) != "":
                base = string_of(raw.get(field))
                break
        self.base = base
        self.key_env = string_of(raw.get("key_env"))
        self.key = string_of(raw.get("key"))


class RatingRow:
    __slots__ = ("model", "provider", "rating_milli", "context_length",
                 "tools", "vision", "json", "free")

    def __init__(self, raw):
        self.model = string_of(raw.get("model"))
        self.provider = string_of(raw.get("provider"))
        rating = number_or_none(raw.get("rating_milli"))
        self.rating_milli = rating if rating is not None else 0
        self.context_length = number_or_none(raw.get("context_length"))
        self.tools = raw.get("tools") is True
        self.vision = raw.get("vision") is True
        self.json = raw.get("json") is True
        self.free = raw.get("free") is True


class Brief:
    def __init__(self):
        self.providers = []
        self.ratings = {}          # (provider, model) -> RatingRow
        self.manifest = []         # workload rows in arrival order
        self.error = None


def read_brief(path):
    brief = Brief()
    if path == "":
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        brief.error = "cannot read %s: %s" % (path, err)
        return brief
    docs = documents_in(text)
    directory = None
    ratings_doc = None
    workload_doc = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory = doc
        elif doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
        elif doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            workload_doc = doc
    if directory is None:
        brief.error = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in directory["providers"]:
        if not is_record(raw):
            continue
        provider = Provider(raw)
        if provider.id != "" and provider.wire in ("openai", "anthropic", "gemini"):
            brief.providers.append(provider)
    if ratings_doc is not None:
        for raw in ratings_doc["ratings"]:
            if not is_record(raw):
                continue
            row = RatingRow(raw)
            if row.model != "" and row.provider != "":
                brief.ratings[(row.provider, row.model)] = row
    if workload_doc is not None:
        for raw in workload_doc["manifest"]:
            if not is_record(raw) or string_of(raw.get("id")) == "":
                continue
            requires = raw.get("requires") if is_record(raw.get("requires")) else {}
            entry = {
                "id": string_of(raw.get("id")),
                "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                "stream": raw.get("stream") is True,
                "requires": {
                    "tools": requires.get("tools") is True,
                    "vision": requires.get("vision") is True,
                    "json": requires.get("json") is True,
                    "min_context": number_or_none(requires.get("min_context")),
                },
            }
            brief.manifest.append(entry)
    return brief


# --------------------------------------------------------------------------
# HTTP plumbing (urllib, sequential, deterministic).
# --------------------------------------------------------------------------

def join_base(env_url, base):
    b = base.strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(provider):
    if provider.key == "" or provider.auth == "none":
        return {}
    if provider.auth == "bearer":
        return {"Authorization": "Bearer " + provider.key}
    if provider.auth == "x-api-key":
        return {"x-api-key": provider.key}
    if provider.auth == "x-goog-api-key":
        return {"x-goog-api-key": provider.key}
    return {}


def http_request(method, url, headers=None, body=None, timeout=CALL_TIMEOUT_S):
    """One HTTP call. Returns (status, headers dict, body bytes).

    Never raises for an HTTP error status; raises OSError/URLError only for
    transport failures, which callers treat as a provider failure.
    """
    data = None
    final_headers = dict(headers or {})
    if body is not None:
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        final_headers.setdefault("content-type", "application/json")
    request = urllib.request.Request(url, data=data, method=method, headers=final_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers.items()) if err.headers else {}, err.read()


# --------------------------------------------------------------------------
# Catalog probe: each provider's model list, read in its own wire.
# --------------------------------------------------------------------------

class ProbedModel:
    __slots__ = ("id", "display_name", "free", "context_length",
                 "tools", "vision", "json")

    def __init__(self, model_id):
        self.id = model_id
        self.display_name = model_id
        self.free = None
        self.context_length = None
        self.tools = None
        self.vision = None
        self.json = None


def zero_priced(record):
    """0 when every price in a listing's pricing object is zero, else None."""
    pricing = record.get("pricing")
    if not is_record(pricing):
        return None
    values = []
    for value in pricing.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, str) and value.strip() != "":
            try:
                values.append(float(value))
            except ValueError:
                pass
    if not values:
        return None
    return all(v == 0 for v in values)


def model_in(wire, raw):
    if not is_record(raw):
        return None
    if wire == "gemini":
        model_id = string_of(raw.get("name"))
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
    else:
        model_id = string_of(raw.get("id"))
    if model_id == "":
        return None
    model = ProbedModel(model_id)
    display = string_of(raw.get("display_name")) or string_of(raw.get("displayName"))
    if display != "":
        model.display_name = display
    if isinstance(raw.get("free"), bool):
        model.free = raw.get("free")
    else:
        model.free = zero_priced(raw)
    for field in ("context_length", "context_window", "inputTokenLimit"):
        value = number_or_none(raw.get(field))
        if value is not None:
            model.context_length = value
            break
    for field in ("tools", "vision", "json"):
        if isinstance(raw.get(field), bool):
            setattr(model, field, raw.get(field))
    return model


def probe_provider(env_url, provider):
    """(ok, [ProbedModel]) for one provider's catalog route."""
    base = join_base(env_url, provider.base)
    if provider.wire == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider))
    if provider.wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    try:
        status, _headers, raw = http_request("GET", url, headers=headers,
                                             timeout=PROBE_TIMEOUT_S)
    except Exception as err:  # transport failure: provider stays out
        log("probe %s failed: %s" % (provider.id, err))
        return False, []
    if status < 200 or status >= 300:
        log("probe %s answered HTTP %s" % (provider.id, status))
        return False, []
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False, []
    if not is_record(body):
        return False, []
    listing = body.get("models") if provider.wire == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return False, []
    models = []
    for raw_model in listing:
        model = model_in(provider.wire, raw_model)
        if model is not None:
            models.append(model)
    return True, models


# --------------------------------------------------------------------------
# The catalog: the --list document and the register body are the same object.
# --------------------------------------------------------------------------

class Candidate:
    """One routable (provider, model) pair with its resolved facts."""

    __slots__ = ("provider", "model_id", "free", "context_length", "tools",
                 "vision", "json", "rating_milli")

    def __init__(self, provider, model, row):
        self.provider = provider
        self.model_id = model.id
        if row is not None:
            self.free = row.free
            self.context_length = row.context_length
            self.tools = row.tools
            self.vision = row.vision
            self.json = row.json
            self.rating_milli = row.rating_milli
        else:
            self.free = model.free if model.free is not None else False
            self.context_length = model.context_length
            self.tools = model.tools
            self.vision = model.vision
            self.json = model.json
            self.rating_milli = 0

    def meets(self, requires):
        if requires.get("tools") and self.tools is not True:
            return False
        if requires.get("vision") and self.vision is not True:
            return False
        if requires.get("json") and self.json is not True:
            return False
        minimum = requires.get("min_context")
        if minimum is not None:
            if self.context_length is None or self.context_length < minimum:
                return False
        return True


class Directory:
    """Probes + catalog + routing tables, built once from the brief."""

    def __init__(self, brief, env_url):
        self.brief = brief
        self.catalog = {"providers": []}
        self.candidates = []       # every probed (provider, model)
        self.free_ids = []         # sorted unique free model ids served
        if env_url == "":
            return
        free_seen = set()
        for provider in brief.providers:
            ok, models = probe_provider(env_url, provider)
            if not ok:
                continue
            entries = []
            seen = set()
            for model in sorted(models, key=lambda m: m.id):
                if model.id in seen:
                    continue
                seen.add(model.id)
                row = brief.ratings.get((provider.id, model.id))
                candidate = Candidate(provider, model, row)
                self.candidates.append(candidate)
                entries.append({
                    "id": model.id,
                    "free": candidate.free,
                    "context_length": candidate.context_length,
                })
                if candidate.free:
                    free_seen.add(model.id)
            self.catalog["providers"].append({
                "id": provider.id,
                "base_url": provider.base,
                "key_env": provider.key_env,
                "wire": provider.wire,
                "models": entries,
            })
        self.free_ids = sorted(free_seen)

    def route_order(self, requires, client_api, json_mode_expected):
        """Candidates for one request, best first.

        Order: rating desc; then the client's own wire (a pass-through keeps
        every echoed field exact); then brief provider order; then model id.
        An Anthropic-wire provider can never echo an OpenAI JSON-mode flag,
        so it is excluded when the client asked for JSON mode on that wire.
        """
        client_wire = "openai" if client_api == "openai" else "anthropic"
        provider_rank = {p.id: i for i, p in enumerate(self.brief.providers)}
        chosen = []
        for candidate in self.candidates:
            if not candidate.free:
                continue
            if not candidate.meets(requires):
                continue
            if json_mode_expected and candidate.provider.wire == "anthropic":
                continue
            chosen.append(candidate)
        chosen.sort(key=lambda c: (
            -c.rating_milli,
            0 if c.provider.wire == client_wire else 1,
            provider_rank.get(c.provider.id, 99),
            c.model_id,
        ))
        return chosen


# --------------------------------------------------------------------------
# Canonical request: the echo-critical fields of a client request, parsed
# once from whichever wire the client used.
#
#   system_text   the system prompt as one string ("" when absent)
#   messages      [{role: user|assistant|tool, parts: [...], name?, tool_call_id?}]
#                 parts: {"kind": "text", "text": str}
#                        {"kind": "image", "media": str, "data": str}   base64
#                        {"kind": "image_url", "url": str}
#                        {"kind": "tool_call", "id": str, "name": str, "input": obj}
#                        {"kind": "tool_result", "tool_use_id": str, "text": str}
#   tools         [{name, description, parameters}]
#   max_tokens    number or None (absence is echoed too: never invent one)
#   stops         [str]
#   json_mode     True when the client asked for a JSON object/schema
#   temperature, top_p, tool_choice (canonical: "auto"|"required"|{"name": x}|None)
# --------------------------------------------------------------------------

def text_of_blocks(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def parse_openai_request(body):
    canon = {
        "system_text": "",
        "messages": [],
        "tools": [],
        "max_tokens": None,
        "stops": [],
        "json_mode": False,
        "temperature": number_or_none(body.get("temperature")),
        "top_p": number_or_none(body.get("top_p")),
        "tool_choice": None,
    }
    system_parts = []
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_record(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = text_of_blocks(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "tool":
            canon["messages"].append({
                "role": "tool",
                "tool_call_id": string_of(message.get("tool_call_id")),
                "name": string_of(message.get("name")),
                "parts": [{"kind": "text",
                           "text": content if isinstance(content, str) else text_of_blocks(content)}],
            })
            continue
        if role not in ("user", "assistant"):
            continue
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"kind": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"kind": "text", "text": block["text"]})
                elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                    url = string_of(block["image_url"].get("url"))
                    parts.append({"kind": "image_ref", "url": url})
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            for call in message["tool_calls"]:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                arguments = call["function"].get("arguments")
                parsed_args = {}
                if isinstance(arguments, str) and arguments.strip() != "":
                    try:
                        value = json.loads(arguments)
                        if is_record(value):
                            parsed_args = value
                    except ValueError:
                        parsed_args = {}
                elif is_record(arguments):
                    parsed_args = arguments
                parts.append({
                    "kind": "tool_call",
                    "id": string_of(call.get("id")),
                    "name": string_of(call["function"].get("name")),
                    "input": parsed_args,
                })
        canon["messages"].append({"role": role, "parts": parts})
    canon["system_text"] = "\n".join(system_parts)
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    for tool in tools:
        if is_record(tool) and is_record(tool.get("function")):
            fn = tool["function"]
            if isinstance(fn.get("name"), str):
                canon["tools"].append({
                    "name": fn["name"],
                    "description": string_of(fn.get("description")),
                    "parameters": fn.get("parameters") if is_record(fn.get("parameters"))
                    else {"type": "object", "properties": {}},
                })
    max_tokens = number_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = number_or_none(body.get("max_completion_tokens"))
    canon["max_tokens"] = max_tokens
    stop = body.get("stop")
    if isinstance(stop, str):
        canon["stops"] = [stop]
    elif isinstance(stop, list):
        canon["stops"] = [s for s in stop if isinstance(s, str)]
    response_format = body.get("response_format")
    if is_record(response_format) and response_format.get("type") in ("json_object", "json_schema"):
        canon["json_mode"] = True
        canon["response_format"] = response_format
    choice = body.get("tool_choice")
    if choice == "auto":
        canon["tool_choice"] = "auto"
    elif choice == "required":
        canon["tool_choice"] = "required"
    elif is_record(choice) and is_record(choice.get("function")):
        canon["tool_choice"] = {"name": string_of(choice["function"].get("name"))}
    return canon


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of_blocks(content)
        return text if text != "" else json.dumps(content)
    if content is None:
        return ""
    return json.dumps(content)


def parse_anthropic_request(body):
    canon = {
        "system_text": text_of_blocks(body.get("system")),
        "messages": [],
        "tools": [],
        "max_tokens": number_or_none(body.get("max_tokens")),
        "stops": [],
        "json_mode": False,
        "temperature": number_or_none(body.get("temperature")),
        "top_p": number_or_none(body.get("top_p")),
        "tool_choice": None,
    }
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_record(message):
            continue
        role = "assistant" if message.get("role") == "assistant" else "user"
        content = message.get("content")
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"kind": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                kind = block.get("type")
                if kind == "text" and isinstance(block.get("text"), str):
                    parts.append({"kind": "text", "text": block["text"]})
                elif kind == "image" and is_record(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        parts.append({
                            "kind": "image",
                            "media": string_of(source.get("media_type")) or "image/png",
                            "data": source["data"],
                        })
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        parts.append({"kind": "image_ref", "url": source["url"]})
                elif kind == "tool_use":
                    parts.append({
                        "kind": "tool_call",
                        "id": string_of(block.get("id")),
                        "name": string_of(block.get("name")),
                        "input": block.get("input") if is_record(block.get("input")) else {},
                    })
                elif kind == "tool_result":
                    parts.append({
                        "kind": "tool_result",
                        "tool_use_id": string_of(block.get("tool_use_id")),
                        "text": tool_result_text(block.get("content")),
                    })
        canon["messages"].append({"role": role, "parts": parts})
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    for tool in tools:
        if is_record(tool) and isinstance(tool.get("name"), str):
            canon["tools"].append({
                "name": tool["name"],
                "description": string_of(tool.get("description")),
                "parameters": tool.get("input_schema") if is_record(tool.get("input_schema"))
                else {"type": "object", "properties": {}},
            })
    stops = body.get("stop_sequences")
    if isinstance(stops, list):
        canon["stops"] = [s for s in stops if isinstance(s, str)]
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            canon["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            canon["tool_choice"] = "required"
        elif choice.get("type") == "tool":
            canon["tool_choice"] = {"name": string_of(choice.get("name"))}
    return canon


# --------------------------------------------------------------------------
# Canonical -> upstream wire bodies.
# --------------------------------------------------------------------------

def split_data_url(url):
    """(media_type, base64 data) for a data: URL, else None."""
    if not url.startswith("data:"):
        return None
    header, sep, data = url.partition(",")
    if sep == "" or ";base64" not in header:
        return None
    media = header[len("data:"):].split(";")[0] or "image/png"
    return media, data


def openai_content(parts):
    """OpenAI message content for canonical parts: a plain string when the
    parts are one text run, else a part list (both parse to the same echo)."""
    blocks = []
    for part in parts:
        if part["kind"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif part["kind"] == "image":
            blocks.append({"type": "image_url", "image_url": {
                "url": "data:%s;base64,%s" % (part["media"], part["data"])}})
        elif part["kind"] == "image_ref":
            blocks.append({"type": "image_url", "image_url": {"url": part["url"]}})
    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return blocks[0]["text"]
    return blocks


def build_openai_body(canon, model):
    messages = []
    if canon["system_text"] != "":
        messages.append({"role": "system", "content": canon["system_text"]})
    for message in canon["messages"]:
        if message["role"] == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": message.get("tool_call_id", ""),
                "content": "".join(p["text"] for p in message["parts"] if p["kind"] == "text"),
            })
            continue
        parts = message["parts"]
        tool_calls = [p for p in parts if p["kind"] == "tool_call"]
        content_parts = [p for p in parts if p["kind"] != "tool_call"]
        entry = {"role": message["role"]}
        if content_parts:
            entry["content"] = openai_content(content_parts)
        else:
            entry["content"] = None
        if tool_calls:
            entry["tool_calls"] = [{
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"],
                             "arguments": json.dumps(call["input"])},
            } for call in tool_calls]
        messages.append(entry)
    body = {"model": model, "messages": messages}
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        body["top_p"] = canon["top_p"]
    if canon["stops"]:
        body["stop"] = canon["stops"]
    if canon["tools"]:
        body["tools"] = [{
            "type": "function",
            "function": {"name": tool["name"],
                         "description": tool["description"],
                         "parameters": tool["parameters"]},
        } for tool in canon["tools"]]
    choice = canon["tool_choice"]
    if choice == "auto":
        body["tool_choice"] = "auto"
    elif choice == "required":
        body["tool_choice"] = "required"
    elif is_record(choice):
        body["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if "response_format" in canon:
        body["response_format"] = canon["response_format"]
    return body


def build_anthropic_body(canon, model):
    messages = []
    for message in canon["messages"]:
        if message["role"] == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": "".join(p["text"] for p in message["parts"] if p["kind"] == "text"),
            }]})
            continue
        blocks = []
        for part in message["parts"]:
            if part["kind"] == "text":
                blocks.append({"type": "text", "text": part["text"]})
            elif part["kind"] == "image":
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": part["media"], "data": part["data"]}})
            elif part["kind"] == "image_ref":
                data_url = split_data_url(part["url"])
                if data_url is not None:
                    blocks.append({"type": "image", "source": {
                        "type": "base64", "media_type": data_url[0], "data": data_url[1]}})
                else:
                    blocks.append({"type": "image", "source": {
                        "type": "url", "url": part["url"]}})
            elif part["kind"] == "tool_call":
                blocks.append({"type": "tool_use", "id": part["id"],
                               "name": part["name"], "input": part["input"]})
            elif part["kind"] == "tool_result":
                blocks.append({"type": "tool_result",
                               "tool_use_id": part["tool_use_id"],
                               "content": part["text"]})
        messages.append({"role": message["role"], "content": blocks})
    body = {"model": model, "messages": messages}
    if canon["system_text"] != "":
        body["system"] = canon["system_text"]
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        body["top_p"] = canon["top_p"]
    if canon["stops"]:
        body["stop_sequences"] = canon["stops"]
    if canon["tools"]:
        body["tools"] = [{
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        } for tool in canon["tools"]]
    choice = canon["tool_choice"]
    if choice == "auto":
        body["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        body["tool_choice"] = {"type": "any"}
    elif is_record(choice):
        body["tool_choice"] = {"type": "tool", "name": choice["name"]}
    return body


def build_gemini_body(canon, model):
    contents = []
    for message in canon["messages"]:
        if message["role"] == "tool":
            contents.append({"role": "user", "parts": [{
                "functionResponse": {
                    "name": message.get("name", ""),
                    "response": {"result": "".join(
                        p["text"] for p in message["parts"] if p["kind"] == "text")},
                }}]})
            continue
        parts = []
        for part in message["parts"]:
            if part["kind"] == "text":
                parts.append({"text": part["text"]})
            elif part["kind"] == "image":
                parts.append({"inlineData": {"mimeType": part["media"],
                                             "data": part["data"]}})
            elif part["kind"] == "image_ref":
                data_url = split_data_url(part["url"])
                if data_url is not None:
                    parts.append({"inlineData": {"mimeType": data_url[0],
                                                 "data": data_url[1]}})
                else:
                    parts.append({"fileData": {"fileUri": part["url"]}})
            elif part["kind"] == "tool_call":
                parts.append({"functionCall": {"name": part["name"],
                                               "args": part["input"]}})
            elif part["kind"] == "tool_result":
                parts.append({"functionResponse": {
                    "name": "",
                    "response": {"result": part["text"]}}})
        contents.append({
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": parts,
        })
    body = {"contents": contents}
    if canon["system_text"] != "":
        body["systemInstruction"] = {"parts": [{"text": canon["system_text"]}]}
    if canon["tools"]:
        body["tools"] = [{"functionDeclarations": [{
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        } for tool in canon["tools"]]}]
    choice = canon["tool_choice"]
    if choice is not None:
        mode = "AUTO" if choice == "auto" else "ANY" if choice == "required" else "ANY"
        config = {"mode": mode}
        if is_record(choice):
            config["allowedFunctionNames"] = [choice["name"]]
        body["toolConfig"] = {"functionCallingConfig": config}
    generation = {}
    if canon["max_tokens"] is not None:
        generation["maxOutputTokens"] = canon["max_tokens"]
    if canon["stops"]:
        generation["stopSequences"] = canon["stops"]
    if canon["temperature"] is not None:
        generation["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        generation["topP"] = canon["top_p"]
    if canon["json_mode"]:
        generation["responseMimeType"] = "application/json"
    if generation:
        body["generationConfig"] = generation
    return body


# --------------------------------------------------------------------------
# Upstream answers, parsed into one internal shape.
# --------------------------------------------------------------------------

def parse_openai_answer(body):
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    if not choices or not is_record(choices[0]):
        return None
    message = choices[0].get("message")
    if not is_record(message):
        return None
    content = message.get("content")
    if isinstance(content, str):
        text = content
    else:
        text = text_of_blocks(content)
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {
        "text": text,
        "input": number_or_none(usage.get("prompt_tokens")) or 0,
        "output": number_or_none(usage.get("completion_tokens")) or 0,
    }


def parse_anthropic_answer(body):
    content = body.get("content") if isinstance(body.get("content"), list) else []
    text = "".join(
        block["text"] for block in content
        if is_record(block) and block.get("type") == "text"
        and isinstance(block.get("text"), str))
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {
        "text": text,
        "input": number_or_none(usage.get("input_tokens")) or 0,
        "output": number_or_none(usage.get("output_tokens")) or 0,
    }


def parse_gemini_answer(body):
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
    if not candidates or not is_record(candidates[0]):
        return None
    content = candidates[0].get("content") if is_record(candidates[0].get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(part["text"] for part in parts
                   if is_record(part) and isinstance(part.get("text"), str))
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return {
        "text": text,
        "input": number_or_none(usage.get("promptTokenCount")) or 0,
        "output": number_or_none(usage.get("candidatesTokenCount")) or 0,
    }


def call_provider(env_url, candidate, canon):
    """One upstream call. (answer, status): answer None on any refusal."""
    provider = candidate.provider
    base = join_base(env_url, provider.base)
    headers = dict(auth_headers(provider))
    if provider.wire == "openai":
        url = base + "/v1/chat/completions"
        payload = build_openai_body(canon, candidate.model_id)
        parser = parse_openai_answer
    elif provider.wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = ANTHROPIC_VERSION
        payload = build_anthropic_body(canon, candidate.model_id)
        parser = parse_anthropic_answer
    else:
        url = (base + "/v1beta/models/" +
               urllib.parse.quote(candidate.model_id, safe="") +
               ":generateContent")
        payload = build_gemini_body(canon, candidate.model_id)
        parser = parse_gemini_answer
    try:
        status, _headers, raw = http_request(
            "POST", url, headers=headers,
            body=json.dumps(payload).encode("utf-8"))
    except Exception as err:
        log("call %s/%s failed: %s" % (provider.id, candidate.model_id, err))
        return None, 0
    if status < 200 or status >= 300:
        return None, status
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, status
    if not is_record(body):
        return None, status
    answer = parser(body)
    if answer is None:
        return None, status
    return answer, status


# --------------------------------------------------------------------------
# Renderers: one internal answer -> the client's wire, JSON or SSE.
# The provider's stop is always the natural-stop family; it is reported in
# the client's own spelling ("stop" / "end_turn").
# --------------------------------------------------------------------------

def split_text(text, size=24):
    pieces = [text[i:i + size] for i in range(0, len(text), size)]
    return pieces if pieces else [""]


def render_openai_json(answer, model):
    return {
        "id": "chatcmpl-aggregate",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer["text"]},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": answer["input"],
            "completion_tokens": answer["output"],
            "total_tokens": answer["input"] + answer["output"],
        },
    }


def sse_data(payload):
    return "data: %s\n\n" % json.dumps(payload)


def render_openai_sse(answer, model):
    def chunk(delta, finish):
        return {
            "id": "chatcmpl-aggregate",
            "object": "chat.completion.chunk",
            "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    frames = [sse_data(chunk({"role": "assistant", "content": ""}, None))]
    for piece in split_text(answer["text"]):
        frames.append(sse_data(chunk({"content": piece}, None)))
    frames.append(sse_data(chunk({}, "stop")))
    frames.append(sse_data({
        "id": "chatcmpl-aggregate",
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": answer["input"],
            "completion_tokens": answer["output"],
            "total_tokens": answer["input"] + answer["output"],
        },
    }))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def render_anthropic_json(answer, model):
    return {
        "id": "msg_aggregate",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": answer["text"]}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": answer["input"],
                  "output_tokens": answer["output"]},
    }


def sse_event(event, payload):
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(payload))


def render_anthropic_sse(answer, model):
    frames = [sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_aggregate",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": answer["input"], "output_tokens": 0},
        },
    })]
    frames.append(sse_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in split_text(answer["text"]):
        frames.append(sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(sse_event("content_block_stop", {
        "type": "content_block_stop", "index": 0}))
    frames.append(sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": answer["output"]}}))
    frames.append(sse_event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_error(message, error_type, code=None):
    return {"error": {"message": message, "type": error_type, "code": code}}


def anthropic_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


# --------------------------------------------------------------------------
# The server.
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, brief, directory):
        self.brief = brief
        self.directory = directory
        self.manifest_index = 0
        self.lock = threading.Lock()
        self.provider_failures = {}

    def requirements_for(self):
        manifest = self.brief.manifest
        entry = manifest[self.manifest_index] if self.manifest_index < len(manifest) else None
        self.manifest_index += 1
        if entry is not None:
            return entry["requires"]
        return {"tools": False, "vision": False, "json": False, "min_context": None}

    def models_payload(self, anthropic_shape):
        ids = self.directory.free_ids
        if anthropic_shape:
            return {
                "data": [{
                    "type": "model",
                    "id": model_id,
                    "display_name": model_id,
                    "created_at": FIXED_CREATED_AT,
                } for model_id in ids],
                "has_more": False,
            }
        return {
            "object": "list",
            "data": [{
                "id": model_id,
                "object": "model",
                "created": FIXED_CREATED,
                "owned_by": "aggregator",
            } for model_id in ids],
        }

    def serve(self, api, body):
        """(status, payload, is_sse) for one client request."""
        if api == "openai":
            canon = parse_openai_request(body)
        else:
            canon = parse_anthropic_request(body)
        requires = self.requirements_for()
        stream = body.get("stream") is True
        candidates = self.directory.route_order(requires, api, canon["json_mode"])
        for position, candidate in enumerate(candidates):
            answer, status = call_provider(ENV_URL, candidate, canon)
            if answer is None and position == 0 and status in (402, 429):
                # A one-off refusal on the provider's failure schedule costs
                # the best model if we move on at once; a spent budget or an
                # outage refuses the retry too, and we fail over then. Only
                # the provider's first failure of the trial earns a retry.
                provider_id = candidate.provider.id
                seen = self.provider_failures.get(provider_id, 0)
                if seen == 0:
                    answer, status = call_provider(ENV_URL, candidate, canon)
            if answer is None:
                provider_id = candidate.provider.id
                self.provider_failures[provider_id] = \
                    self.provider_failures.get(provider_id, 0) + 1
                continue
            model = candidate.model_id
            if api == "openai":
                payload = render_openai_sse(answer, model) if stream else render_openai_json(answer, model)
            else:
                payload = render_anthropic_sse(answer, model) if stream else render_anthropic_json(answer, model)
            return 200, payload, stream
        if api == "openai":
            return 503, openai_error("no free model could serve this request",
                                     "server_error", "no_free_model"), False
        return 503, anthropic_error("overloaded_error",
                                    "no free model could serve this request"), False


def make_handler(aggregator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, status, payload, is_sse=False):
            if isinstance(payload, (dict, list)):
                raw = json.dumps(payload).encode("utf-8")
                content_type = "application/json"
            else:
                raw = payload.encode("utf-8")
                content_type = "text/event-stream"
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(raw)))
            if is_sse:
                self.send_header("cache-control", "no-cache")
            self.end_headers()
            self.wfile.write(raw)

        def _error(self, api, status, message):
            if api == "anthropic":
                self._send(status, anthropic_error("invalid_request_error", message))
            else:
                self._send(status, openai_error(message, "invalid_request_error"))

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/v1/models":
                anthropic_shape = self.headers.get("x-api-key") is not None
                self._send(200, aggregator.models_payload(anthropic_shape))
                return
            self._error("openai", 404, "no route GET %s" % path)

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/v1/chat/completions":
                api = "openai"
            elif path == "/v1/messages":
                api = "anthropic"
            else:
                self._error("openai", 404, "no route POST %s" % path)
                return
            length = self.headers.get("content-length")
            try:
                size = int(length) if length is not None else 0
            except ValueError:
                size = 0
            if size > MAX_BODY_BYTES:
                self._error(api, 413, "the request body is too large")
                return
            raw = self.rfile.read(size) if size > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else None
            except (ValueError, UnicodeDecodeError):
                body = None
            if not is_record(body):
                self._error(api, 400, "the request body is not one JSON object")
                return
            with aggregator.lock:
                status, payload, is_sse = aggregator.serve(api, body)
            self._send(status, payload, is_sse)

    return Handler


def register(port, catalog):
    status, _headers, raw = http_request(
        "POST", ENV_URL + "/register",
        body=json.dumps({
            "protocol_version": PROTOCOL_VERSION,
            "port": port,
            "catalog": catalog,
        }).encode("utf-8"),
        timeout=PROBE_TIMEOUT_S)
    if status < 200 or status >= 300:
        raise RuntimeError("POST /register answered HTTP %s: %s"
                           % (status, raw[:500]))
    reply = json.loads(raw.decode("utf-8"))
    if is_record(reply) and isinstance(reply.get("requests"), int):
        log("registered: the workload holds %s requests" % reply["requests"])


def poll_until_done(server):
    while True:
        try:
            status, _headers, raw = http_request("GET", ENV_URL + "/work",
                                                 timeout=2.0)
            if status == 200:
                reply = json.loads(raw.decode("utf-8"))
                if is_record(reply) and reply.get("done") is True:
                    break
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_S)
    threading.Thread(target=server.shutdown, daemon=True).start()


def main(argv):
    brief = read_brief(PROMPT_FILE)
    if brief.error is not None:
        log(brief.error)
    directory = Directory(brief, ENV_URL)
    if "--list" in argv:
        text = json.dumps(directory.catalog, separators=(",", ":")) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0
    if ENV_URL == "":
        log("MAKEFUN_ENV_URL is not set")
        return 1
    aggregator = Aggregator(brief, directory)
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(aggregator))
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout")
        return 1
    log("listening on 0.0.0.0:%s with %s providers and %s free models"
        % (port, len(directory.catalog["providers"]), len(directory.free_ids)))
    register(port, directory.catalog)
    poller = threading.Thread(target=poll_until_done, args=(server,), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

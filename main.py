import asyncio  # Async event loop for MCP client
import json  # Parse/format tool arguments and results
import time  # For Unix timestamps in LibreChat response
import os  # Environment variables and paths
import sys  # Detect stdin mode for LibreChat single-shot
from typing import Any, Dict, List  # Type hints

from dotenv import load_dotenv  # Load .env
from openai import OpenAI  # OpenAI SDK


# MCP stdio client pieces
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# FastAPI HTTP server pieces (for LibreChat HTTP mode)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel


# Define the MCP stdio server launch params (runs mcp_mongo_server.py)
SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,  # Use current interpreter to avoid 'python' not found
    args=["mcp_mongo_server.py"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    env=dict(os.environ),
)


# OpenAI tool specs that mirror MCP tools
OPENAI_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_profile",
            "description": "Upsert Mustafa Asghari profile and TDEE",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "gender": {"type": "string"},
                    "weight_kg": {"type": "number"},
                    "height_cm": {"type": "number"},
                    "tdee": {"type": "number"},
                },
                "required": ["age", "gender", "weight_kg", "height_cm", "tdee"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_profile",
            "description": "Fetch a user by name (default 'Mustafa Asghari')",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "default": "Mustafa Asghari"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "meal_add",
            "description": "Add a meal with calories, ingredients, and nutrition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "calories": {"type": "number"},
                    "ingredients": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "protein_g": {"type": "number", "default": 0},
                    "carbs_g": {"type": "number", "default": 0},
                    "fat_g": {"type": "number", "default": 0},
                    "healthy_fat_g": {"type": "number", "default": 0},
                    "unhealthy_fat_g": {"type": "number", "default": 0},
                },
                "required": ["name", "calories", "ingredients"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "meals_list",
            "description": "List meals for a date (default: today)",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "meal_delete",
            "description": "Delete meals by name for a date (default: today)",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "total_get",
            "description": "Get daily total calories and nutrition for a date (default: today)",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
            },
        },
    },
]


def format_librechat_response(response) -> Dict[str, Any]:
    """Return exactly what LibreChat expects for a chat completion."""
    return {
        "id": getattr(response, "id", f"chatcmpl-{int(time.time())}"),
        "object": getattr(response, "object", "chat.completion"),
        "created": getattr(response, "created", int(time.time())),
        "model": getattr(response, "model", "gpt-4o-mini"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.choices[0].message.content or "",
                },
                "finish_reason": response.choices[0].finish_reason or "stop",
            }
        ],
        "usage": response.usage.model_dump() if hasattr(response, "usage") and response.usage else None,
    }


async def call_mcp_tool(session: ClientSession, name: str, arguments: Dict[str, Any]) -> str:
    """Call an MCP tool and return its JSON string result."""
    result = await session.call_tool(name, arguments or {})
    if result is None:
        return json.dumps({"result": None})
    try:
        payload = result.to_dict()  # Prefer dict conversion if available
    except Exception:
        payload = {
            "content": [
                c.to_dict() if hasattr(c, "to_dict") else str(c)
                for c in getattr(result, "content", [])
            ]
        }
    return json.dumps(payload)


async def chat_loop() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment/.env")

    client = OpenAI(api_key=api_key)

    async def run_with_messages(messages: List[Dict[str, Any]], model: str = "gpt-4o-mini", quiet: bool = False):
        # Start MCP stdio server and session
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if not quiet:
                    print("MCP server connected. Type 'exit' to quit.")

                # Tool-call loop until the model stops requesting tools
                while True:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=OPENAI_TOOLS,
                        tool_choice="auto",
                    )
                    msg = resp.choices[0].message

                    if msg.tool_calls:
                        # Append the assistant message that contains tool_calls (required by OpenAI API)
                        messages.append(
                            {
                                "role": "assistant",
                                "content": msg.content or "",
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for tc in msg.tool_calls
                                ],
                            }
                        )

                        # Execute each tool call and append corresponding tool messages
                        for tc in msg.tool_calls:
                            name = tc.function.name
                            args = tc.function.arguments
                            try:
                                parsed = json.loads(args) if isinstance(args, str) else args
                            except Exception:
                                parsed = {}

                            # Forward the tool call to MCP
                            tool_output_json = await call_mcp_tool(session, name, parsed)

                            # Feed tool result back to the model
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "name": name,
                                    "content": tool_output_json,
                                }
                            )
                        continue  # Give the model a turn with tool outputs

                    return resp  # Final response

    # If called by LibreChat as a one-shot process, read JSON from stdin and return once
    is_lc_env = os.getenv("LIBRECHAT_MODE") == "1"
    stdin_data = None
    if not sys.stdin.isatty():
        try:
            stdin_raw = sys.stdin.read().strip()
            if stdin_raw:
                stdin_data = json.loads(stdin_raw)
        except Exception:
            stdin_data = None

    if is_lc_env or stdin_data:
        # Expect OpenAI-compatible request body: { model, messages }
        body = stdin_data or {}
        model = body.get("model", "gpt-4o-mini")
        req_messages = body.get("messages", [])
        # Prepend our system prompt to steer tool usage
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are Calorie Assistant for Mustafa Asghari. Use tools to store/retrieve "
                    "profile, meals, nutrition, and daily totals when needed."
                ),
            }
        ]
        # Append provided messages
        for m in req_messages:
            messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        resp = await run_with_messages(messages, model=model, quiet=True)
        # Output exactly what LibreChat expects and exit
        print(json.dumps(format_librechat_response(resp)))
        return

    # Interactive REPL fallback
    base_messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are Calorie Assistant for Mustafa Asghari. When appropriate, "
                "decide to call tools to store/retrieve profile, meals, nutrition, and totals."
            ),
        }
    ]
    print("MCP server connected. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! bro 😭")
            break

        if user_input.lower() in {"exit", "quit", "q", "bye", "goodbye"}:
            print("Goodbye! bro 😭")
            break

        convo = base_messages + [{"role": "user", "content": user_input}]
        resp = await run_with_messages(convo, model="gpt-4o-mini", quiet=False)
        # Also print LibreChat JSON for visibility
        print(json.dumps(format_librechat_response(resp)))


if __name__ == "__main__":
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print("\nGoodbye! bro 😭")


# =========================
# FastAPI HTTP proxy (OpenAI-compatible)
# =========================

class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = "gpt-4o-mini"
    messages: List[ChatMessage]
    stream: bool | None = False


app = FastAPI(title="OpenAI-compatible MCP proxy in main.py")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def http_chat_completions(req: ChatCompletionRequest):
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing in env")

        client = OpenAI(api_key=api_key)

        # Build initial messages with our system prompt
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are Calorie Assistant for Mustafa Asghari. Use tools to store/retrieve "
                    "profile, meals, nutrition, and daily totals when needed."
                ),
            }
        ]
        for m in req.messages:
            messages.append({"role": m.role, "content": m.content or ""})

        # Start MCP stdio server and process tool calls for this request
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                while True:
                    resp = client.chat.completions.create(
                        model=req.model or "gpt-4o-mini",
                        messages=messages,
                        tools=OPENAI_TOOLS,
                        tool_choice="auto",
                    )
                    msg = resp.choices[0].message

                    if msg.tool_calls:
                        # Append assistant with tool_calls per OpenAI protocol
                        messages.append(
                            {
                                "role": "assistant",
                                "content": msg.content or "",
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for tc in msg.tool_calls
                                ],
                            }
                        )

                        # Execute tools and append tool results
                        for tc in msg.tool_calls:
                            name = tc.function.name
                            args = tc.function.arguments
                            try:
                                parsed = json.loads(args) if isinstance(args, str) else args
                            except Exception:
                                parsed = {}
                            tool_output_json = await call_mcp_tool(session, name, parsed)
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "name": name,
                                    "content": tool_output_json,
                                }
                            )
                        continue

                    # Final answer: return strict JSON with exact Content-Type and no trailing bytes
                    payload = format_librechat_response(resp)
                    return Response(
                        content=json.dumps(payload, separators=(",", ":")),  # minified, no trailing spaces
                        media_type="application/json",  # exactly application/json
                    )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"ERROR in http_chat_completions: {error_msg}", file=sys.stderr)
        print(f"TRACEBACK: {error_traceback}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": error_msg,
                    "type": type(e).__name__,
                    "traceback": error_traceback,
                }
            }
        )
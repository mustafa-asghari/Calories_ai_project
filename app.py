import os
import time
import json
import traceback
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession


SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["mcp_mongo_server.py"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    env=dict(os.environ),
)

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
                "properties": {"name": {"type": "string", "default": "Mustafa Asghari"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "meal_add",
            "description": "Add a meal with calories, ingredients, and nutrition. ALWAYS call meals_list() first to check for duplicates.",
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
            "description": "List ALL meals for a date (default: today). CRITICAL: Always call this BEFORE inserting, updating, or deleting to check what exists.",
            "parameters": {"type": "object", "properties": {"date": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "meal_delete",
            "description": "Delete meals by name for a date (default: today). You have FULL PERMISSION to delete any meal.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "date": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "meals_delete_all",
            "description": "Delete ALL meals for a date (default: today). FULL PERMISSION: You can delete all meals for a date. Use when you need to clear all data and start fresh.",
            "parameters": {"type": "object", "properties": {"date": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "meal_update",
            "description": "Update an existing meal. FULL PERMISSION: You can update any meal. Always call meals_list() first to verify the meal exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "calories": {"type": "number"},
                    "ingredients": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                    "protein_g": {"type": "number"},
                    "carbs_g": {"type": "number"},
                    "fat_g": {"type": "number"},
                    "healthy_fat_g": {"type": "number"},
                    "unhealthy_fat_g": {"type": "number"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "total_get",
            "description": "Get daily total calories and nutrition for a date (default: today). ALWAYS call this to verify totals match expected calculations.",
            "parameters": {"type": "object", "properties": {"date": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "total_delete",
            "description": "Delete the daily total entry for a date (default: today). FULL PERMISSION: You can delete daily totals. Use when totals are incorrect due to duplicate entries.",
            "parameters": {"type": "object", "properties": {"date": {"type": "string"}}},
        },
    },
]


async def call_mcp_tool(session: ClientSession, name: str, arguments: Dict[str, Any]) -> str:
    result = await session.call_tool(name, arguments or {})
    if result is None:
        return json.dumps({"result": None})
    try:
        payload = result.to_dict()
    except Exception:
        payload = {
            "content": [
                c.to_dict() if hasattr(c, "to_dict") else str(c)
                for c in getattr(result, "content", [])
            ]
        }
    return json.dumps(payload)


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    
    class Config:
        # Allow extra fields for compatibility with LibreChat
        extra = "allow"


class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = "gpt-4o-mini"  # Make model optional with default
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    stream: Optional[bool] = False
    
    class Config:
        # Allow extra fields for compatibility with LibreChat
        extra = "allow"


# Default models available (LibreChat compatibility)
DEFAULT_MODELS = [
    {
        "id": "gpt-4o-mini",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "ai-calorie",
    },
    {
        "id": "gpt-4o",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "ai-calorie",
    },
    {
        "id": "gpt-4-turbo",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "ai-calorie",
    },
]


load_dotenv()
app = FastAPI(title="AI Calorie OpenAI-compatible API for LibreChat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add request logging middleware to track all incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging"""
    import time
    start_time = time.time()
    
    # Log request details
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    url = str(request.url)
    user_agent = request.headers.get("user-agent", "unknown")
    
    print(f"[REQUEST] {method} {url} - IP: {client_ip} - User-Agent: {user_agent[:50]}", file=sys.stderr)
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    print(f"[RESPONSE] {method} {url} - Status: {response.status_code} - Time: {process_time:.3f}s", file=sys.stderr)
    
    return response


# Add exception handler for validation errors to return OpenAI-compatible format
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors in OpenAI-compatible format"""
    errors = exc.errors()
    error_messages = [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in errors]
    error_message = "; ".join(error_messages)
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": f"Invalid request: {error_message}",
                "type": "invalid_request_error",
                "code": "invalid_request"
            }
        }
    )


def get_api_key(authorization: Optional[str] = Header(None)) -> str:
    """
    Validate API key from Authorization header (for LibreChat compatibility).
    LibreChat sends: Authorization: Bearer <api_key> or Authorization: <api_key>
    We accept any value (including "dummy") for LibreChat compatibility,
    but we always use OPENAI_API_KEY from environment for actual OpenAI calls.
    """
    # Accept any API key from the request (LibreChat may send "dummy")
    # We don't actually use this for OpenAI calls - we use OPENAI_API_KEY from env
    if authorization:
        # Remove "Bearer " prefix if present
        if authorization.startswith("Bearer "):
            api_key = authorization[7:]
        else:
            api_key = authorization
        # Accept any API key value (including "dummy")
        return api_key or "accepted"
    
    # If no Authorization header, that's also fine for LibreChat
    return "accepted"


@app.get("/v1/models")
async def list_models(request: Request, authorization: Optional[str] = Header(None)):
    """
    List available models (required by LibreChat).
    LibreChat calls this endpoint to discover available models.
    This endpoint must respond quickly (LibreChat has a 5 second timeout).
    """
    # Log the request for debugging
    print(f"GET /v1/models - IP: {request.client.host if request.client else 'unknown'}, Headers: {dict(request.headers)}", file=sys.stderr)
    
    # Return immediately - no processing needed
    response = {
        "object": "list",
        "data": DEFAULT_MODELS
    }
    print(f"GET /v1/models - Returning {len(DEFAULT_MODELS)} models", file=sys.stderr)
    return response


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionsRequest,
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """
    Chat completions endpoint compatible with LibreChat and OpenAI API.
    Supports tool calling, streaming, and proper authentication.
    """
    # Log the request for debugging
    print(f"POST /v1/chat/completions - IP: {request.client.host if request.client else 'unknown'}, Model: {payload.model}", file=sys.stderr)
    
    try:
        # Normalize model name (handle variations like "gpt-4o-mini-2024-07-18" -> "gpt-4o-mini")
        model = payload.model or "gpt-4o-mini"
        if model and "-2024-07-18" in model:
            model = model.replace("-2024-07-18", "")
        if model not in [m["id"] for m in DEFAULT_MODELS]:
            # Use gpt-4o-mini as default if model not found
            model = "gpt-4o-mini"
        
        # Always use OPENAI_API_KEY from environment, NOT the api_key from request
        # The api_key parameter is only for LibreChat compatibility (may be "dummy")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable is not set. Please configure it in Railway environment variables."
            )
        
        client = OpenAI(api_key=openai_api_key)

        # Build messages list, preserving tool call history and tool message fields
        messages: List[Dict[str, Any]] = []
        
        # Add system message that defines the AI's role and purpose
        # Check if there's already a system message in the payload
        # payload.messages contains ChatMessage Pydantic objects, so access .role attribute directly
        has_system_message = any(m.role == "system" for m in payload.messages)
        
        if not has_system_message:
            # Add system message at the beginning if not present
            # This defines the AI's role as a fitness assistant for Mustafa Asghari with detailed database guidance
            system_message = {
                "role": "system",
                "content": """You are an AI assistant created by Mustafa Asghari to help him in his fitness journey. You help track calories, meals, nutrition, and provide fitness guidance.

CRITICAL DATABASE OPERATION RULES:

1. ALWAYS CHECK THE DATABASE FIRST:
   - Before inserting ANY meal, ALWAYS call meals_list(date) to see what meals already exist for that date
   - Before updating or deleting, ALWAYS verify what's in the database first
   - This prevents duplicates and ensures accurate calculations

2. NEVER CREATE "Daily Total" ENTRIES AS MEALS:
   - Daily totals are AUTOMATICALLY calculated in the daily_totals collection
   - DO NOT create meal entries named "Daily Total", "Total", or similar
   - Only create actual meal entries (e.g., "Tuna Rice Bowl", "Protein Bar")

3. CALCULATION ACCURACY:
   - When calculating totals, manually sum: calories, protein_g, carbs_g, fat_g, healthy_fat_g, unhealthy_fat_g
   - ALWAYS verify calculations: fat_g MUST equal healthy_fat_g + unhealthy_fat_g
   - After any insert/update/delete, call total_get(date) to verify the database totals match your calculations
   - If totals are wrong, check meals_list() for duplicate entries or incorrect data

4. FIXING INCORRECT DATA:
   - If totals are doubled or incorrect, first call meals_list(date) to see ALL entries
   - Delete any duplicate meals or incorrect "Daily Total" meal entries using meal_delete()
   - If needed, delete all meals for a date using meals_delete_all() and start fresh
   - You have FULL PERMISSION to delete and update anything - use it when data is wrong

5. DATABASE STRUCTURE:
   - meals collection: Individual meal entries (name, calories, protein_g, carbs_g, fat_g, healthy_fat_g, unhealthy_fat_g)
   - daily_totals collection: Automatically calculated totals (total_calories, total_protein_g, total_carbs_g, total_fat_g, total_healthy_fat_g, total_unhealthy_fat_g)
   - Daily totals are recalculated automatically when meals are added/updated/deleted

6. WORKFLOW FOR UPDATING TOTALS:
   - Step 1: Call meals_list(date) to see current meals
   - Step 2: Identify and delete any incorrect/duplicate entries
   - Step 3: Add or update correct meal entries
   - Step 4: Call total_get(date) to verify totals match expected calculations
   - Step 5: If totals are still wrong, check meals_list() again for remaining issues

You have FULL PERMISSION to insert, update, and delete everything. Be proactive in fixing data issues. Always verify calculations and database state before and after operations."""
            }
            messages.append(system_message)
            print("Added enhanced system message with database operation guidance", file=sys.stderr)
        else:
            print("System message already present in request, keeping existing message", file=sys.stderr)
        
        # Add all messages from the request
        for m in payload.messages:
            msg_dict = m.model_dump(exclude_none=True)
            # Ensure all fields are preserved (tool_calls, tool_call_id, name for tool messages)
            messages.append(msg_dict)

        # CRITICAL: This Railway API MUST ALWAYS use MCP server for database access
        # The Railway API's purpose is to provide database access through tools
        # We ALWAYS use tools/MCP - this is not optional for database functionality
        # 
        # Flow: LibreChat -> Railway API -> OpenAI API (with tools) -> MCP Server -> MongoDB
        # The Railway API acts as a middleware that adds database tools to all requests
        
        print("Railway API: Ensuring database access through MCP server and tools", file=sys.stderr)
        
        # Handle streaming: LibreChat requests streaming, but tool calls need to complete first
        # We'll get the full response (with tool calls) then stream it back to LibreChat
        requested_stream = payload.stream if payload.stream else False
        if requested_stream:
            print("Streaming requested - will get full response with tools then stream it back", file=sys.stderr)

        # Tool calling path with MCP
        # MCP server provides database access through tools - this is REQUIRED for database functionality
        # We MUST use MCP to access MongoDB, so we don't fall back to OpenAI-only responses
        print("Starting MCP server for database access...", file=sys.stderr)
        
        # Verify MONGODB_URI is set before starting MCP
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            print("WARNING: MONGODB_URI not set - MCP server may fail", file=sys.stderr)
        else:
            print(f"MONGODB_URI is set (length: {len(mongodb_uri)})", file=sys.stderr)
        
        # Initialize MCP session and handle errors properly
        mcp_init_error = None
        try:
            async with stdio_client(SERVER_PARAMS) as (read, write):
                async with ClientSession(read, write) as session:
                    try:
                        print("Initializing MCP session...", file=sys.stderr)
                        await session.initialize()
                        print("MCP session initialized successfully - database access available", file=sys.stderr)
                    except Exception as init_error:
                        import traceback
                        error_tb = traceback.format_exc()
                        print(f"ERROR: MCP session initialization failed: {init_error}", file=sys.stderr)
                        print(f"MCP init traceback: {error_tb}", file=sys.stderr)
                        # Store error to handle after exiting context manager
                        mcp_init_error = str(init_error)
                        # Re-raise to exit context managers cleanly
                        raise
                    
                    # MCP session initialized successfully - proceed with tool calling

                    while True:
                        # ALWAYS use tools for database access - this is the Railway API's main purpose
                        # Use tools from request if provided, otherwise use our database tools
                        tools = payload.tools if payload.tools is not None else OPENAI_TOOLS
                        # Force tool_choice to "auto" to ensure tools are available for database access
                        # (unless explicitly set to "none" by the client for non-database requests)
                        tool_choice = payload.tool_choice if payload.tool_choice is not None else "auto"
                        
                        print(f"Using tools for database access: {len(tools)} tools available, tool_choice={tool_choice}", file=sys.stderr)
                        
                        try:
                            # Always disable streaming when using tools (checked earlier)
                            resp = client.chat.completions.create(
                                model=model,
                                messages=messages,
                                tools=tools,
                                tool_choice=tool_choice,
                                temperature=payload.temperature,
                                max_tokens=payload.max_tokens,
                                stream=False,  # Streaming disabled for tool calls
                            )
                        except Exception as openai_error:
                            print(f"OpenAI API call failed: {openai_error}", file=sys.stderr)
                            raise
                        
                        choice = resp.choices[0]
                        msg = choice.message

                        if msg.tool_calls:
                            # Append the assistant message with tool_calls
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

                            # Execute each tool call
                            for tc in msg.tool_calls:
                                name = tc.function.name
                                args = tc.function.arguments
                                try:
                                    parsed = json.loads(args) if isinstance(args, str) else args
                                except Exception as parse_error:
                                    print(f"Failed to parse tool arguments: {parse_error}", file=sys.stderr)
                                    parsed = {}

                                try:
                                    tool_output_json = await call_mcp_tool(session, name, parsed)
                                    print(f"Tool {name} executed successfully", file=sys.stderr)
                                except Exception as tool_error:
                                    print(f"Tool {name} execution failed: {tool_error}", file=sys.stderr)
                                    import traceback
                                    print(f"Tool error traceback: {traceback.format_exc()}", file=sys.stderr)
                                    # Return error as tool output so the model can handle it
                                    tool_output_json = json.dumps({
                                        "error": str(tool_error),
                                        "type": type(tool_error).__name__
                                    })

                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tc.id,
                                        "name": name,
                                        "content": tool_output_json,
                                    }
                                )
                            continue  # Loop back to get the model's response to tool outputs

                        # Final response - include tool_calls in message if present (LibreChat compatibility)
                        created = int(time.time())
                        response_message: Dict[str, Any] = {
                            "role": "assistant",
                            "content": msg.content or "",
                        }
                        
                        # Include tool_calls if present (important for LibreChat)
                        if msg.tool_calls:
                            response_message["tool_calls"] = [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in msg.tool_calls
                            ]
                        
                        # If streaming was requested, stream the final response back to LibreChat
                        # NOTE: This code is only reached when we have a FINAL response (no more tool calls)
                        # Tool calls are executed in the loop above, and we continue until we get a final answer
                        if requested_stream:
                            print("Streaming final response to LibreChat (after all tool calls completed)", file=sys.stderr)
                            def generate_stream_from_response():
                                # Stream content in chunks to match OpenAI streaming format
                                content = response_message.get("content", "") or ""
                                
                                # First chunk: send role
                                role_chunk = {
                                    "id": resp.id or f"chatcmpl-{created}",
                                    "object": "chat.completion.chunk",
                                    "created": resp.created or created,
                                    "model": resp.model or model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"role": "assistant"},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(role_chunk)}\n\n"
                                
                                # Stream content in word-sized chunks for better UX
                                if content:
                                    words = content.split(" ")
                                    for i, word in enumerate(words):
                                        if word:  # Skip empty strings
                                            is_last = i == len(words) - 1
                                            chunk_data = {
                                                "id": resp.id or f"chatcmpl-{created}",
                                                "object": "chat.completion.chunk",
                                                "created": resp.created or created,
                                                "model": resp.model or model,
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": {
                                                            "content": word + (" " if not is_last else "")
                                                        },
                                                        "finish_reason": choice.finish_reason if is_last else None,
                                                    }
                                                ],
                                            }
                                            yield f"data: {json.dumps(chunk_data)}\n\n"
                                
                                # Final chunk with finish reason
                                if not content or choice.finish_reason:
                                    finish_chunk = {
                                        "id": resp.id or f"chatcmpl-{created}",
                                        "object": "chat.completion.chunk",
                                        "created": resp.created or created,
                                        "model": resp.model or model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": choice.finish_reason or "stop",
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(finish_chunk)}\n\n"
                                
                                yield "data: [DONE]\n\n"
                            
                            return StreamingResponse(
                                generate_stream_from_response(),
                                media_type="text/event-stream"
                            )
                        
                        # Non-streaming response
                        return {
                            "id": resp.id or f"chatcmpl-{created}",
                            "object": "chat.completion",
                            "created": resp.created or created,
                            "model": resp.model or model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": response_message,
                                    "finish_reason": choice.finish_reason or "stop",
                                }
                            ],
                            "usage": resp.usage.model_dump() if resp.usage else None,
                        }
        except Exception as mcp_error:
            # MCP server is REQUIRED for database access - we cannot fall back to OpenAI-only
            # Log the error with full details for debugging
            import traceback
            error_traceback = traceback.format_exc()
            error_message = str(mcp_error)
            error_type = type(mcp_error).__name__
            
            print(f"CRITICAL: MCP Error in chat_completions: {error_message}", file=sys.stderr)
            print(f"MCP Error traceback: {error_traceback}", file=sys.stderr)
            
            # Check if it's a TaskGroup/ExceptionGroup error and extract underlying error
            underlying_error = error_message
            if "TaskGroup" in error_type or "ExceptionGroup" in error_message:
                # Try to extract the underlying error from nested exceptions
                if hasattr(mcp_error, "__cause__") and mcp_error.__cause__:
                    underlying_error = str(mcp_error.__cause__)
                    print(f"Underlying error (from __cause__): {underlying_error}", file=sys.stderr)
                elif hasattr(mcp_error, "exceptions") and mcp_error.exceptions:
                    # Extract from ExceptionGroup
                    try:
                        underlying_error = str(mcp_error.exceptions[0])
                        print(f"Underlying error (from exceptions[0]): {underlying_error}", file=sys.stderr)
                    except (IndexError, AttributeError):
                        pass
            
            # Use stored initialization error if available (cleaner error message)
            if mcp_init_error:
                underlying_error = mcp_init_error
                print(f"Using stored MCP init error: {underlying_error}", file=sys.stderr)
            
            # Check for common MCP issues
            if "MONGODB_URI" in underlying_error or "mongodb" in underlying_error.lower():
                error_hint = "MongoDB connection issue - check MONGODB_URI environment variable"
            elif "mcp_mongo_server" in underlying_error.lower() or "import" in underlying_error.lower():
                error_hint = "MCP server startup issue - check mcp_mongo_server.py and dependencies"
            else:
                error_hint = "MCP server error - check Railway logs for details"
            
            # Return error - database access is required, cannot proceed without it
            error_detail = {
                "error": {
                    "message": f"Database access unavailable: {underlying_error}. {error_hint}",
                    "type": error_type,
                    "code": "database_unavailable",
                    "hint": "The MCP server provides database access. Please ensure MONGODB_URI is set and MCP server can start."
                }
            }
            
            return JSONResponse(
                status_code=500,
                content=error_detail
            )
    except HTTPException:
        raise
    except Exception as e:
        # Always include traceback in error for debugging
        import traceback
        error_traceback = traceback.format_exc()
        error_message = str(e)
        error_type = type(e).__name__
        
        # Log the error for debugging
        print(f"ERROR in chat_completions: {error_message}", file=sys.stderr)
        print(f"TRACEBACK: {error_traceback}", file=sys.stderr)
        
        # Return OpenAI-compatible error format that LibreChat understands
        # OpenAI error format: {"error": {"message": "...", "type": "...", "code": "..."}}
        error_detail = {
            "error": {
                "message": error_message,
                "type": error_type,
                "code": "internal_error"
            }
        }
        
        # Return JSONResponse with proper error format
        return JSONResponse(
            status_code=500,
            content=error_detail
        )


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "AI Calorie Tracker - LibreChat Compatible",
        "endpoints": [
            "GET /v1/models",
            "POST /v1/chat/completions",
            "GET /health"
        ]
    }


@app.get("/health")
async def health():
    """
    Health check endpoint for Railway and monitoring.
    Returns basic status without requiring MCP server to start.
    """
    status: Dict[str, Any] = {
        "ok": True,
        "has_OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "has_MONGODB_URI": bool(os.getenv("MONGODB_URI")),
        "mcp_start_ok": False,
    }
    # Don't fail health check if MCP can't start - just report it
    try:
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                status["mcp_start_ok"] = True
    except Exception as e:
        # Log but don't fail - MCP will start when needed
        status["mcp_error"] = str(e)[:100]  # Limit error message length
    return status

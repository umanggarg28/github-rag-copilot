"""Test Gemma 4 tool calling via Gemini API before switching."""
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the indexed codebase for relevant code snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_symbol",
            "description": "Find a specific function, class, or variable by name.",
            "parameters": {
                "type": "object",
                "properties": {"symbol_name": {"type": "string"}},
                "required": ["symbol_name"],
            },
        },
    },
]

messages = [
    {"role": "system", "content": "You are a code research assistant. Always call search_code or search_symbol before answering. Never answer from memory."},
    {"role": "user", "content": "How does the backward pass work for a multiplication operation?"},
]

for model in ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]:
    print(f"\n{'='*55}\nTesting {model}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=512,
        )
        c = response.choices[0]
        print(f"finish_reason: {c.finish_reason}")
        print(f"tool_calls: {len(c.message.tool_calls or [])}")
        if c.message.tool_calls:
            for tc in c.message.tool_calls:
                print(f"  → {tc.function.name}({tc.function.arguments})")
            print("✅ PASS")
        else:
            print(f"  content: {(c.message.content or '')[:150]}")
            print("❌ FAIL — no tool calls")
    except Exception as e:
        print(f"❌ ERROR: {e}")

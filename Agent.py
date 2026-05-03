from openai import OpenAI
from tools import tavily_search
import json
import os
import json
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

tools = [
    {
        "type": "function",
        "name": "tavily_search",
        "description": "Search the web for up-to-date information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results"
                }
            },
            "required": ["query"]
        }
    }
]

# ✅ Step 4 — Initial request
response = client.responses.create(
    model="gpt-4o",
    tools=tools,
    input="Find the latest news about AI"
)

# ✅ Step 5 — Tool loop
while True:
    tool_called = False

    for item in response.output:
        print(f"Item type: {item.type}, Status: {getattr(item, 'status', 'N/A')}")

        if item.type == "function_call":
            tool_called = True

            if item.name == "tavily_search":
                args = json.loads(item.arguments)
                print(f"Calling tavily_search with: {args}")
                result = tavily_search(**args)
                print(f"Tool result: {result}")

                response = client.responses.create(
                    model="gpt-4o",
                    tools=tools,
                    previous_response_id=response.id,   # ⭐ important
                    input=[
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps(result)
                        }
                    ]
                )
                print(f"New response output: {response.output}")

    if not tool_called:
        break

# ✅ Final output
print("\n=== FINAL OUTPUT ===")
print("Response output:", response.output)
print("Response output_text:", response.output_text)
print("\nFinal answer:")
for item in response.output:
    if item.type == "text":
        print(item.text)
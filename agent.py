"""
Question-Answering Agent with Tool Calling (Local LLM Version)

This module implements the core chatbot logic using a local Llama-3 model.
"""

import json
import re
import math
import os
import sys
from typing import Any, List, Dict

# Import llama-cpp-python
try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed. Please run: pip install llama-cpp-python")
    sys.exit(1)

# Initialize the Local LLM
# We look for the model in the 'models' directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")

if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model not found at {MODEL_PATH}")
    print("Please run 'python setup_local.py' to download the model.")
    llm = None
else:
    print(f"Loading local model from {MODEL_PATH}...")
    # n_ctx=4096 is the context window size
    # n_gpu_layers=-1 attempts to offload all layers to GPU if available
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=-1, 
        verbose=False
    )
    print("Model loaded successfully!")

# Define our tools
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Get the current time and date.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

def execute_web_search(query: str) -> str:
    """Simulates a web search."""
    # Mock search results
    mock_results = {
        "weather": "Current weather: Sunny, 72°F (22°C). Humidity 45%.",
        "stock": "Stock Update: TSLA $210.50 (+1.2%), AAPL $185.00 (-0.5%).",
        "news": "Latest News: AI advancements continue to accelerate. New local models released.",
        "default": f"Search results for '{query}': Found multiple relevant pages about {query}."
    }
    
    query_lower = query.lower()
    if "weather" in query_lower:
        return mock_results["weather"]
    elif "stock" in query_lower or "price" in query_lower:
        return mock_results["stock"]
    elif "news" in query_lower:
        return mock_results["news"]
    else:
        return mock_results["default"]


def execute_get_current_time() -> str:
    """Returns the current date and time."""
    from datetime import datetime
    return f"The current date and time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def execute_calculator(expression: str) -> str:
    """Safely evaluates mathematical expressions."""
    try:
        expr = expression.strip()
        # Handle "X% of Y"
        percent_match = re.match(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)', expr, re.IGNORECASE)
        if percent_match:
            percent = float(percent_match.group(1))
            value = float(percent_match.group(2))
            return f"{percent}% of {value} = {(percent/100)*value}"
        
        expr = expr.replace('^', '**').replace('×', '*').replace('÷', '/')
        safe_dict = {
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log10, 'ln': math.log, 'exp': math.exp, 'abs': abs,
            'pow': pow, 'pi': math.pi, 'e': math.e
        }
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "web_search":
        return execute_web_search(tool_input.get("query", ""))
    elif tool_name == "calculator":
        return execute_calculator(tool_input.get("expression", ""))
    elif tool_name == "get_current_time":
        return execute_get_current_time()
    else:
        return f"Unknown tool: {tool_name}"


def chat(user_message: str, conversation_history: list = None) -> dict:
    if conversation_history is None:
        conversation_history = []
    
    # Check if model is loaded
    if llm is None:
        return {
            "response": "Error: Local model not loaded. Please check server logs.",
            "tool_calls": [],
            "conversation_history": conversation_history
        }

    # Add user message
    conversation_history.append({"role": "user", "content": user_message})
    
    # System prompt to guide the model to use tools via JSON
    # System prompt to guide the model to use tools via JSON
    system_prompt = """You are a QA Agent with access to tools.
You MUST use the 'web_search' tool for ANY question about current events, weather, stock prices, or real-time information.
You MUST use the 'calculator' tool for ANY math problem.
You MUST use the 'get_current_time' tool for ANY question about what time or date it is.

Tools available:
1. web_search: Input: {"query": "string"}
2. calculator: Input: {"expression": "string"}
3. get_current_time: Input: {}

IMPORTANT: To use a tool, you must output ONLY a valid JSON object. Do not add any other text.
Format:
{
  "tool": "tool_name",
  "input": { ... }
}

Example:
User: What is the weather in London?
Assistant: { "tool": "web_search", "input": { "query": "weather in London" } }

If no tool is needed, just answer normally.
"""
    
    # Prepare messages for Llama
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    
    # 1. Ask Model
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )
    
    content = response["choices"][0]["message"]["content"]
    tool_calls_made = []
    
    # 2. Check for tool usage (JSON pattern)
    tool_json = None
    try:
        # Look for a JSON block or just try parsing the whole content
        # Clean specific markdown code blocks if present
        clean_content = content.strip()
        if "```json" in clean_content:
            clean_content = clean_content.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_content:
            clean_content = clean_content.split("```")[1].split("```")[0].strip()
            
        if clean_content.startswith("{") and "tool" in clean_content:
            tool_json = json.loads(clean_content)
    except Exception:
        pass # Not a valid JSON tool call
    
    # 3. Handle Tool Call
    if tool_json and "tool" in tool_json:
        tool_name = tool_json["tool"]
        tool_input = tool_json.get("input", {})
        
        # Record the tool call
        tool_calls_made.append({
            "tool": tool_name,
            "input": tool_input,
            "output": None # Will fill later
        })
        
        # Execute
        result = execute_tool(tool_name, tool_input)
        tool_calls_made[-1]["output"] = result
        
        # Add assistant's *intent* to use tool to history
        conversation_history.append({"role": "assistant", "content": json.dumps(tool_json)})
        
        # Add tool result to history
        conversation_history.append({
            "role": "function", # Or user, but function is more semantic if supported? Llama-3 standard is usually user or specific tool role if using native tools.
            # Using 'user' role with 'Tool Output:' prefix is robust for standard chat models
            "content": f"Tool Output for {tool_name}: {result}" 
        })
        
        # Ask Model again with result
        messages = [{"role": "system", "content": system_prompt}] + conversation_history
        response_2 = llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        final_response = response_2["choices"][0]["message"]["content"]
        
        conversation_history.append({"role": "assistant", "content": final_response})
        
    else:
        # No tool used
        final_response = content
        conversation_history.append({"role": "assistant", "content": final_response})
    
    return {
        "response": final_response,
        "tool_calls": tool_calls_made,
        "conversation_history": conversation_history
    }


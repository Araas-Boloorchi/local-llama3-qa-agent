"""
FastAPI Server for the Question-Answering Agent

This server provides REST API endpoints for:
1. /chat - Send messages and get responses from the chatbot
2. /evaluate - Run the evaluation suite and get results
3. /evaluation-results - Get the latest evaluation results

HOW IT WORKS:
- FastAPI is a modern Python web framework for building APIs
- It uses async/await for handling concurrent requests
- Pydantic models define the request/response schemas
- CORS middleware allows the frontend to communicate with this backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import json
import os

# Import our agent and evaluation modules
from agent import chat
from evaluation import run_evaluation, get_evaluation_results

# Create the FastAPI application
app = FastAPI(
    title="QA Agent API",
    description="API for the Question-Answering Agent with tool calling",
    version="1.0.0"
)

# Configure CORS (Cross-Origin Resource Sharing)
# This is necessary for the frontend (running on a different port) to call our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# ============================================================================
# PYDANTIC MODELS
# These define the structure of request/response data
# Pydantic automatically validates incoming data against these schemas
# ============================================================================

class ChatRequest(BaseModel):
    """
    Request body for the /chat endpoint.
    
    - message: The user's question or input
    - session_id: Optional ID to maintain conversation context
    """
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """
    Response from the /chat endpoint.
    
    - response: The chatbot's answer
    - tool_calls: List of tools that were used (for transparency)
    - session_id: The session ID for this conversation
    """
    response: str
    tool_calls: list
    session_id: str


class EvaluationResponse(BaseModel):
    """
    Response from the /evaluate endpoint.
    
    - results: List of individual test results
    - summary: Aggregate metrics (accuracy, avg score, etc.)
    """
    results: list
    summary: dict


# ============================================================================
# IN-MEMORY SESSION STORAGE
# In production, you'd use Redis or a database
# This stores conversation history so users can have multi-turn conversations
# ============================================================================

sessions = {}


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """
    Serve the frontend HTML file.
    """
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/api")
async def api_info():
    """
    API info endpoint.
    """
    return {
        "message": "QA Agent API is running",
        "version": "1.0.0",
        "endpoints": ["/chat", "/evaluate", "/evaluation-results"]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint - send a message, get a response.
    
    How it works:
    1. Check if this is an existing session or create a new one
    2. Retrieve conversation history for context
    3. Call the agent with the message and history
    4. Store the updated history
    5. Return the response
    
    The conversation history allows for follow-up questions like:
    User: "What is the capital of France?"
    Agent: "Paris"
    User: "What's its population?"
    Agent: "Paris has a population of about 2.1 million..."
    """
    try:
        # Generate or use existing session ID
        session_id = request.session_id or f"session_{len(sessions)}"
        
        # Get existing conversation history or start fresh
        conversation_history = sessions.get(session_id, [])
        
        # Call the agent
        result = chat(request.message, conversation_history)
        
        # Store updated conversation history
        sessions[session_id] = result["conversation_history"]
        
        return ChatResponse(
            response=result["response"],
            tool_calls=result["tool_calls"],
            session_id=session_id
        )
    
    except Exception as e:
        # Log the error and return a friendly message
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_endpoint():
    """
    Run the evaluation suite.
    
    This endpoint:
    1. Loads the evaluation dataset (questions with expected answers)
    2. Runs each question through the chatbot
    3. Uses an LLM to judge the quality of responses
    4. Returns detailed results and summary metrics
    
    This is useful for:
    - Measuring chatbot quality
    - Comparing different versions
    - Identifying areas for improvement
    """
    try:
        results = run_evaluation()
        return EvaluationResponse(
            results=results["individual_results"],
            summary=results["summary"]
        )
    except Exception as e:
        print(f"Error in evaluate endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation-results")
async def get_evaluation_results_endpoint():
    """
    Get the most recent evaluation results.
    
    This reads from a stored file so you don't have to re-run
    the evaluation every time you want to see the results.
    """
    try:
        results = get_evaluation_results()
        if results:
            return results
        else:
            return {"message": "No evaluation results found. Run /evaluate first."}
    except Exception as e:
        print(f"Error getting evaluation results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear a conversation session.
    
    Useful when the user wants to start a fresh conversation
    without the context of previous messages.
    """
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


# ============================================================================
# RUN THE SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run on port 3000 to serve both API and frontend
    print("Starting server at http://localhost:3000")
    uvicorn.run(app, host="0.0.0.0", port=3000)

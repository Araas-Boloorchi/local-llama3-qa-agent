# Local Llama-3 QA Agent

> **Note:** This was vacation code and I made it in 6 hours!

This repository contains a full-stack application featuring a local LLM-powered chatbot with tool calling capabilities, an evaluation system, and a web dashboard for interaction and analysis.

## Original Problem Statement

**Goal:** Build an LLM-powered chatbot that interacts with the user on open-ended questions, and then evaluate it for quality.

**Tasks:**
1. Build a simple LLM-based general-purpose conversational question-answering chatbot that makes use of tool calling (function calling).
2. Design and conduct an evaluation of the chatbot (metrics, dataset, and evaluation code).
3. Build a web-based dashboard to display and analyze the evaluation, and interact with the chatbot live.

---

## Project Overview

![QA Agent Demo Video](demo.webp)
This project implements:

1. **Local LLM Chatbot**: A conversational agent powered by Llama-3 (running locally via `llama-cpp-python`) that can use external tools (web search, calculator).
2. **Evaluation System**: An LLM-as-judge evaluation framework to measure chatbot quality.
3. **Web Dashboard**: A React-based interface for chatting and viewing evaluation results.

## Project Structure

```
/
├── agent.py          # Core chatbot logic with tool calling and local LLM
├── server.py         # FastAPI REST API server
├── evaluation.py     # Evaluation dataset and LLM-as-judge
├── setup_local.py    # Script to download the Llama model
├── requirements.txt  # Python dependencies
├── index.html        # Web Dashboard
└── run.py            # Helper script to run the system
```

## Quick Start

### Prerequisites

- Python 3.9+
- 8GB+ RAM recommended (for running the 3B model)
- C++ Compiler (Visual Studio Build Tools on Windows, Xcode on Mac, GCC on Linux) for building `llama-cpp-python`

### 1. Set up the Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download the Model

```bash
# Download the Local Model (approx 2GB)
python setup_local.py
```

### 3. Run the Application

You can use the helper script to verify everything and start the server:

```bash
python run.py
```

Or run the server directly:

```bash
python server.py
```

The application will be available at **http://localhost:3000**

## System Architecture and Interaction

### Local LLM Architecture

```
User Question → Llama-3 (Local) → Tool Call Decision (JSON)
                                     ↓
                             [Execute Tool] ← Tool Result
                                     ↓
                          Llama-3 → Final Response
```

1. **User sends a question** to the chatbot
2. **Llama receives** the question along with available tool definitions in the system prompt
3. **Llama decides** whether to call a tool by outputting a JSON object
4. **Agent executes the tool** and feeds the result back to Llama
5. **Llama generates** the final response

### Available Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `web_search` | Searches the web for information | Current events, real-time data |
| `calculator` | Performs mathematical calculations | Arithmetic, percentages, unit conversions |

### Evaluation Methodology

The evaluation uses an **LLM-as-judge** approach with the local model:

1. **Dataset**: 10 test questions across categories (calculation, factual, search, reasoning)
2. **Execution**: Each question is run through the chatbot
3. **Judging**: Llama evaluates responses on correctness, completeness, and relevance
4. **Metrics**: Results are aggregated into accuracy, average score, and tool usage statistics

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the Web Dashboard |
| `/chat` | POST | Send message, get response |
| `/evaluate` | POST | Run evaluation suite |
| `/evaluation-results` | GET | Get latest results |
| `/session/{id}` | DELETE | Clear a conversation |

## Running Evaluations

You can run evaluations directly from the dashboard or via command line:

```bash
python evaluation.py
```

## Technical Design Decisions

1. **Llama-3 as the LLM**: Chosen for its efficiency and instruction-following capabilities, making it suitable for local execution on consumer hardware.
2. **llama-cpp-python**: Used for running GGUF models efficiently.
3. **LLM-as-judge**: Reused the same local model for evaluation to keep the system self-contained.

## License

MIT License - see LICENSE file for details.

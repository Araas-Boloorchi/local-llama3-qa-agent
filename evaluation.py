"""
Evaluation System for the Question-Answering Agent (Local LLM Version)

Uses local Llama-3 as judge.
"""

import json
import os
import re
from datetime import datetime
from typing import Optional

# Import chat and LLM from agent
from agent import chat, llm

# ============================================================================
# EVALUATION DATASET
# ============================================================================

EVALUATION_DATASET = [
    {
        "id": "math_1",
        "question": "What is 25% of 840?",
        "expected_answer": "210",
        "category": "calculation",
        "requires_tool": "calculator"
    },
    {
        "id": "math_2", 
        "question": "Calculate the square root of 144",
        "expected_answer": "12",
        "category": "calculation",
        "requires_tool": "calculator"
    },
    {
        "id": "math_3",
        "question": "What is 15 * 23 + 45?",
        "expected_answer": "390",
        "category": "calculation",
        "requires_tool": "calculator"
    },
    {
        "id": "factual_1",
        "question": "What is the chemical symbol for gold?",
        "expected_answer": "Au",
        "category": "factual",
        "requires_tool": None
    },
    {
        "id": "factual_2",
        "question": "Who wrote Romeo and Juliet?",
        "expected_answer": "William Shakespeare",
        "category": "factual",
        "requires_tool": None
    },
    {
        "id": "factual_3",
        "question": "What is the capital of Japan?",
        "expected_answer": "Tokyo",
        "category": "factual",
        "requires_tool": None
    },
    {
        "id": "search_1",
        "question": "What's the current weather like?",
        "expected_answer": "Should attempt to search for weather information",
        "category": "search",
        "requires_tool": "web_search"
    },
    {
        "id": "search_2",
        "question": "What are the latest news headlines?",
        "expected_answer": "Should attempt to search for news",
        "category": "search",
        "requires_tool": "web_search"
    },
    {
        "id": "reasoning_1",
        "question": "If a train travels 60 mph for 2.5 hours, how far does it go?",
        "expected_answer": "150 miles",
        "category": "reasoning",
        "requires_tool": "calculator"
    },
    {
        "id": "reasoning_2",
        "question": "A shirt costs $40 and is on sale for 30% off. What's the sale price?",
        "expected_answer": "$28",
        "category": "reasoning",
        "requires_tool": "calculator"
    }
]


def judge_response(question: str, expected: str, actual: str, tool_calls: list) -> dict:
    """
    Use local LLM to judge the quality of a response.
    """
    if llm is None:
        return {
            "correctness": 0, "completeness": 0, "relevance": 0, "overall_score": 0, 
            "explanation": "Judge model not loaded"
        }

    judge_prompt = f"""You are evaluating a chatbot's response to a question.

Question: {question}
Expected Answer: {expected}
Actual Response: {actual}
Tools Used: {json.dumps(tool_calls) if tool_calls else "None"}

Evaluate the response on these criteria:
1. Correctness (1-5): Is the information accurate?
2. Completeness (1-5): Does it fully answer the question?
3. Relevance (1-5): Is the response relevant?

Provide your evaluation in this exact JSON format:
{{
    "correctness": <score>,
    "completeness": <score>,
    "relevance": <score>,
    "overall_score": <average of the three>,
    "explanation": "<brief explanation of your scoring>"
}}

Only output the JSON, nothing else."""

    try:
        messages = [
            {"role": "system", "content": "You are a fair and critical judge."},
            {"role": "user", "content": judge_prompt}
        ]
        
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.1, # deterministic
            max_tokens=500
        )
        
        judge_text = response["choices"][0]["message"]["content"]
        
        # Extract JSON
        if "```json" in judge_text:
            judge_text = judge_text.split("```json")[1].split("```")[0]
        elif "```" in judge_text:
            judge_text = judge_text.split("```")[1].split("```")[0]
            
        evaluation = json.loads(judge_text.strip())
        return evaluation
    
    except Exception as e:
        print(f"Error judging response: {e}")
        return {
            "correctness": 3,
            "completeness": 3,
            "relevance": 3,
            "overall_score": 3,
            "explanation": f"Error during evaluation: {str(e)}"
        }


def run_evaluation() -> dict:
    """Run the full evaluation suite."""
    results = []
    print("Starting evaluation...")
    print(f"Running {len(EVALUATION_DATASET)} test cases\n")
    
    for i, test_case in enumerate(EVALUATION_DATASET):
        print(f"[{i+1}/{len(EVALUATION_DATASET)}] Testing: {test_case['question'][:50]}...")
        
        try:
            chat_result = chat(test_case["question"], conversation_history=None)
            
            evaluation = judge_response(
                question=test_case["question"],
                expected=test_case["expected_answer"],
                actual=chat_result["response"],
                tool_calls=chat_result["tool_calls"]
            )
            
            tools_used = [tc["tool"] for tc in chat_result["tool_calls"]]
            expected_tool = test_case.get("requires_tool")
            correct_tool_used = (expected_tool is None or expected_tool in tools_used)
            
            result = {
                "id": test_case["id"],
                "question": test_case["question"],
                "expected_answer": test_case["expected_answer"],
                "actual_response": chat_result["response"],
                "category": test_case["category"],
                "tools_used": tools_used,
                "expected_tool": expected_tool,
                "correct_tool_used": correct_tool_used,
                "scores": evaluation,
                "passed": evaluation["overall_score"] >= 3.5
            }
            
            results.append(result)
            print(f"   Score: {evaluation['overall_score']}/5 - {'PASS' if result['passed'] else 'FAIL'}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            results.append({
                "id": test_case["id"],
                "question": test_case["question"],
                "error": str(e),
                "passed": False
            })
    
    # Calculate summary statistics
    passed_count = sum(1 for r in results if r.get("passed", False))
    total_count = len(results)
    scores = [r["scores"]["overall_score"] for r in results if "scores" in r]
    avg_score = sum(scores) / len(scores) if scores else 0
    tool_accuracy = sum(1 for r in results if r.get("correct_tool_used", False))
    
    # Category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "scores": []}
        categories[cat]["total"] += 1
        if r.get("passed", False):
            categories[cat]["passed"] += 1
        if "scores" in r:
            categories[cat]["scores"].append(r["scores"]["overall_score"])
    
    for cat in categories:
        s_list = categories[cat]["scores"]
        categories[cat]["avg_score"] = sum(s_list) / len(s_list) if s_list else 0
        categories[cat]["accuracy"] = categories[cat]["passed"] / categories[cat]["total"] if categories[cat]["total"] > 0 else 0
    
    summary = {
        "total_tests": total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "accuracy": passed_count / total_count if total_count > 0 else 0,
        "average_score": avg_score,
        "tool_accuracy": tool_accuracy / total_count if total_count > 0 else 0,
        "category_breakdown": categories,
        "timestamp": datetime.now().isoformat()
    }
    
    output = {"individual_results": results, "summary": summary}
    results_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*50}")
    print("EVALUATION COMPLETE")
    print(f"{'='*50}")
    print(f"Accuracy: {summary['accuracy']*100:.1f}% ({passed_count}/{total_count})")
    print(f"Average Score: {avg_score:.2f}/5")
    print(f"Tool Usage Accuracy: {summary['tool_accuracy']*100:.1f}%")
    print(f"\nResults saved to: {results_path}")
    
    return output


def get_evaluation_results() -> Optional[dict]:
    results_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    run_evaluation()

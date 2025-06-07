# Main script to run the intent classification benchmark against local LLMs.

import requests
import sys
import json
import os
import time
from datetime import datetime

OLLAMA_API_URL = "http://localhost:11434"

def check_ollama_http():
    """
    Verifies that the Ollama HTTP endpoint is reachable.
    If not, prints an error and exits.
    """
    print(f"Verifying Ollama connection at {OLLAMA_API_URL}...")
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        response.raise_for_status()
        print("Ollama connection successful.")
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        print(f"Error: Could not reach Ollama HTTP API. {e}")
        print("Please ensure Ollama is installed and running (e.g., `ollama serve`).")
        sys.exit(1)

def get_available_models_http():
    """Fetches the list of available models from the Ollama API."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        response.raise_for_status()
        data = response.json()
        models = [m["name"] for m in data.get("models", [])]
        print(f"Found {len(models)} available models.")
        return models
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        print(f"Error fetching models: {e}")
        return []

def pull_default_model_cli(model_name="gemma:2b"):
    """
    If no models are installed, falls back to the CLI to pull a default model.
    """
    print(f"No models found. Pulling default model '{model_name}' via CLI...")
    os.system(f"ollama pull {model_name}")
    print(f"Finished pulling {model_name}.")

def run_intent_classification_http(model, system_prompt, user_query):
    """
    Sends a single prompt to the Ollama /api/generate endpoint.
    Returns the JSON response string and the request duration.
    """
    payload = {
        "model": model,
        "prompt": f"{system_prompt}\nUser: {user_query}",
        "stream": False,
        "format": "json",
        "temperature": 0
    }
    
    start_time = time.monotonic()
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60 # Increased timeout for slower models
        )
        response.raise_for_status()
        duration = time.monotonic() - start_time
        data = response.json()
        return data.get("response", "").strip(), duration
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        print(f"Error during API call for model {model}: {e}")
        return None, 0

def main():
    """Main function to run the benchmark."""
    # 1. Ensure Ollama HTTP endpoint is up
    check_ollama_http()

    # 2. Ensure at least one model is available
    models = get_available_models_http()
    if not models:
        pull_default_model_cli()
        models = get_available_models_http()
        if not models:
            print("Error: Still no models available after attempting to pull a default.")
            sys.exit(1)

    # 3. Define the system prompt for intent classification
    system_prompt = (
        "You are an intent classification tool. "
        "Your job is to identify the user's intent behind a query and output the result in strict JSON format "
        "with two keys: 'intent' and 'confidence'.\n\n"

        "Output format:\n"
        '{ "intent": "<one of: weather, time, map, llm, web_search, math, date, unknown>", "confidence": <float between 0 and 1> }\n\n'

        "Intent categories:\n"
        "- 'weather': Questions about current or future weather conditions.\n"
        "- 'time': Requests for current local time in a location or time zone.\n"
        "- 'map': Queries about places, locations, countries, directions, or geography.\n"
        "- 'llm': Questions that can be answered with general world knowledge or historical facts (e.g. capital cities, famous people, events). Do NOT use this intent for recent or changing information.\n"
        "- 'web_search': Questions that require real-time or up-to-date information (e.g. latest news, sports scores, live data, current population or stock prices).\n"
        "- 'math': Requests for numeric calculations or arithmetic.\n"
        "- 'date': Questions about today's date, holidays, or date math (e.g. how many days until...).\n"
        "- 'unknown': Use this if the query is vague, conversational, or doesn't match any known category.\n\n"

        "IMPORTANT:\n"
        "- If a question asks about something recent or dynamic (e.g. 'Who won the Super Bowl?', 'What’s the stock price?'), classify it as 'web_search'.\n"
        "- If a question asks about something timeless or historical (e.g. 'What is the capital of France?', 'Who discovered gravity?'), classify it as 'llm'.\n"
        "- Return ONLY a single JSON object with the predicted intent and confidence score. Do not include explanations, comments, or extra text.\n\n"

        "Examples:\n"
        "User: What is the capital of France?\n"
        '{ "intent": "llm", "confidence": 0.9 }\n\n'
        "User: Who won the last Super Bowl?\n"
        '{ "intent": "web_search", "confidence": 0.8 }\n\n'
    )

    # 4. Define a comprehensive set of test queries
    test_queries = {
        'weather': [
            "Do I need a jacket in Seattle tonight?",
            "What's the wind speed in Chicago?",
            "Will it rain in Houston this weekend?",
            "Is it snowing in Denver right now?",
            "How hot is it in Phoenix today?",
        ],
        'time': [
            "What time is it in Berlin?",
            "Give me the current time in Tokyo.",
            "What's the local time in São Paulo?",
            "Tell me the time in UTC.",
            "Time now in Johannesburg?"
        ],
        'map': [
            "Where is the Eiffel Tower located?",
            "Give me directions to Central Park.",
            "Find the nearest coffee shop to me.",
            "What country is the Great Pyramid of Giza in?",
            "Which continent is Egypt in?"
        ],
        'llm': [  # static facts that LLMs should answer without search
            "What is the capital of France?",
            "Tell me something interesting about ancient Rome.",
            "What year did the Berlin Wall fall?",
            "Who painted the Mona Lisa?",
            "How many continents are there on Earth?"
        ],
        'web_search': [  # dynamic or recent info that should trigger search
            "Search for the latest news on AI.",
            "Find the population of Canada.",
            "Who won the last Super Bowl?",
            "What's the stock price of Nvidia right now?",
            "Search for recent news about electric vehicles."
        ],
        'math': [
            "What's 2+2?",
            "Calculate the square root of 16.",
            "If a car travels 60 miles in 1 hour, how far does it travel in 2.5 hours?",
            "What is 20% of $150?",
            "Multiply 23 by 7."
        ],
        'date': [
            "What's today's date?",
            "Whats the date in New Zealand today?",
            "What day of the week was January 1, 2000?",
            "How many days are there until Christmas?",
            "What’s the date 10 days from now?"
        ]
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"llm_intent_results_{timestamp}.jsonl"

    print(f"\nStarting benchmark... Output will be saved to '{results_filename}'")
    
    # 5. Run tests and save results to a JSONL file
    with open(results_filename, 'a') as f:
        for model in models:
            print(f"\nTesting model: {model}")
            for category, queries in test_queries.items():
                for query in queries:
                    response_text, duration = run_intent_classification_http(model, system_prompt, query)
                    
                    if response_text is None:
                        continue
                    
                    try:
                        parsed = json.loads(response_text)
                        intent = parsed.get('intent', 'unknown')
                        raw_confidence = parsed.get('confidence')
                        confidence = float(raw_confidence) if raw_confidence is not None else 0.0
                        print(f"  Success: '{query}' -> {intent} ({confidence:.2f}) in {duration:.2f}s")
                    except json.JSONDecodeError:
                        print(f"  Failed to parse JSON from model {model} for query '{query}': {response_text}")
                        intent = 'error'
                        confidence = 0.0

                    record = {
                        'model': model,
                        'category': category,
                        'query': query,
                        'intent': intent,
                        'confidence': confidence,
                        'duration': round(duration, 2)
                    }
                    f.write(json.dumps(record) + "\n")

    print(f"\nBenchmark complete. All results saved to: {results_filename}")

    # 6. Automatically call the analysis script on the generated results
    analysis_script = 'analyze_results.py'
    if os.path.exists(analysis_script):
        print(f"\nRunning analysis using '{analysis_script}'...")
        os.system(f"python {analysis_script} {results_filename}")
    else:
        print(f"\nAnalysis script '{analysis_script}' not found. Skipping analysis.")

if __name__ == "__main__":
    main()

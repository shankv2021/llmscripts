'''
This is a sample python script that runs an LLM model using llama-cpp on a public dataset with schema restrictions!

Before executing this script: 

- Do a pip install datasets, tqdm
- Keep the llama-cpp server running in the background. Refer LLAMA-CPP.md file for instructions
- Set LLAMA_MODEL variable to the name of gguf model. Example: For gemma 4b, 
- This script has two ways of output restrictions
  1. GBNF grammar 
  2. JSON schema restriction
'''

import time
import json
import requests
from datasets import load_dataset
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError, RootModel
from typing import Literal


class BoolResponse(RootModel):
    root: Literal[0, 1]

MAX_TOKENS = 1
TEMPERATURE = 0
N_SAMPLES = 200

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"
LLAMA_MODEL = "gemma-3-4b-it" # Name of LLM model
GBNF_GRAMMAR = 'root ::= [01]'

def build_prompt(passage: str, question: str) -> str:
    """Build a prompt that explicitly requests JSON format"""
    return f"""Passage:
{passage}

Question:
{question}

Answer with 1 for true or 0 for false
"""

def call_llama_server(prompt: str, grammar):
    payload = {
        "model": LLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    start = time.perf_counter()
    if grammar:
      payload['grammer'] =  GBNF_GRAMMAR
    else:
      payload["response_format"] = {
          "type": "json_schema",
          "json_schema": {
              "name": "bool_response",
              "schema": BoolResponse.model_json_schema(),
              "strict": True  # Enforce strict schema adherence
          }
      }
    start = time.perf_counter()
    try:
        r = requests.post(LLAMA_SERVER_URL, json=payload, timeout=30)
        r.raise_for_status()
        latency = time.perf_counter() - start
        
        raw = r.json()["choices"][0]["message"]["content"]
        answer = int(raw.strip()) if grammar else BoolResponse.model_validate_json(raw).root 
        return answer, latency, None
    except Exception as e:
        latency = time.perf_counter() - start
        return None, latency, str(e)

def run_experiment(dataset, call_fn, name, grammar):
    """Run experiment for a single backend"""
    print(f"\n{'='*50}")
    print(f"Running {name}")
    print(f"{'='*50}\n")
    
    stats = {"correct": 0, "errors": 0, "latencies": []}
    
    for ex in tqdm(dataset, desc=name):
        gold = 1 if ex["answer"] else 0
        prompt = build_prompt(ex["passage"], ex["question"])
        
        pred, lat, error = call_fn(prompt,grammar)
        stats["latencies"].append(lat)
        
        if error:
            stats["errors"] += 1
            if stats["errors"] <= 3:  # Print first 3 errors
                print(f"\nError: {error}")
        elif pred == gold:
            stats["correct"] += 1
    
    return stats

def print_results(stats, name, total):
    """Print formatted results"""
    acc = stats["correct"] / total * 100
    avg_lat = sum(stats["latencies"]) / len(stats["latencies"])
    error_rate = stats["errors"] / total * 100
    
    print(f"\n{name} Results:")
    print(f"  Accuracy:    {acc:6.2f}%")
    print(f"  Error Rate:  {error_rate:6.2f}%")
    print(f"  Avg Latency: {avg_lat:.4f}s")

dataset = load_dataset("boolq", split="validation")
dataset = dataset.select(range(N_SAMPLES))

start = time.perf_counter()
for ex in tqdm(dataset):
    gold = 1 if ex["answer"] else 0
    prompt = build_prompt(ex["passage"], ex["question"])

    # Test llama-server first
    grammar_restriction_stats = run_experiment(dataset, call_llama_server, "GRAMMAR", grammar=True)
    print_results(grammar_restriction_stats, "GRAMMAR", len(dataset))
    
    # Then test Ollama
    json_schema_restriction_stats = run_experiment(dataset, call_llama_server, "JSON-SCHEMA", grammar=False)
    print_results(json_schema_restriction_stats, "JSON-SCHEMA", len(dataset))
    
    # Final comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print_results(grammar_restriction_stats, "GRAMMAR", len(dataset))
    print_results(json_schema_restriction_stats, "JSON-SCHEMA", len(dataset))))

print(time.perf_counter() - start)

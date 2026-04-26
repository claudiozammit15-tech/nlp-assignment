"""
Gemini API diagnostic script.
Tests the connection, model availability, and response parsing.
"""
import os
from openai import OpenAI

API_KEY = os.environ.get("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

if not API_KEY:
    print("ERROR: GEMINI_API_KEY environment variable not set!")
    exit(1)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- Step 1: List available models ---
print("=" * 60)
print("STEP 1: Listing available Gemini models")
print("=" * 60)
try:
    models = client.models.list()
    for m in models.data:
        print(f"  {m.id}")
except Exception as e:
    print(f"  Could not list models: {e}")

# --- Step 2: Try the configured model name ---
MODEL_NAME = "gemini-2.5-flash"
print()
print("=" * 60)
print(f"STEP 2: Testing model '{MODEL_NAME}'")
print("=" * 60)

test_messages = [{"role": "user", "content": "What is 2+2? Answer in one word."}]

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=test_messages,
        temperature=0.3,
        max_tokens=100
    )
    print(f"  Status: SUCCESS")
    print(f"  Response object type: {type(response)}")
    print(f"  Choices count: {len(response.choices)}")
    for i, choice in enumerate(response.choices):
        print(f"  Choice {i}:")
        print(f"    finish_reason: {choice.finish_reason}")
        print(f"    message.role: {choice.message.role}")
        print(f"    message.content: {repr(choice.message.content)}")
    print(f"  Usage: {response.usage}")
except Exception as e:
    print(f"  FAILED with error: {e}")
    print(f"  Error type: {type(e).__name__}")

# --- Step 3: Try alternative model names ---
print()
print("=" * 60)
print("STEP 3: Testing alternative model names")
print("=" * 60)

alternatives = [
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

working_models = []
for model_name in alternatives:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=test_messages,
            temperature=0.3,
            max_tokens=50
        )
        content = response.choices[0].message.content if response.choices else None
        print(f"  {model_name}: OK -> {repr(content[:80])}")
        working_models.append(model_name)
    except Exception as e:
        print(f"  {model_name}: FAILED -> {e}")

# --- Step 4: Test parse_answer compatibility with a working model ---
if working_models:
    print()
    print("=" * 60)
    print(f"STEP 4: Testing full answer parsing with '{working_models[0]}'")
    print("=" * 60)

    full_prompt = """### Instruction
You are a helpful assistant aiming to answer the following question with reasoning.
Please output in the following format:
```
### Answer
<your step-by-step reasoning>
Final answer: <your final answer>
```
Only choose A,B,C,D,E,F,G,H,I,J representing your choice as your final answer. Please keep your output concise and limit it to 500 words. Use the most short reasoning to get the correct answer. Do not output unrelated content!

### Question
What is 2+2? A) 3 B) 4 C) 5 D) 6
"""

    try:
        response = client.chat.completions.create(
            model=working_models[0],
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3,
            max_tokens=500
        )
        raw = response.choices[0].message.content
        print(f"  Raw response:\n{raw}")
        print()

        # Test parsing
        if "Final answer:" in raw:
            print(f"  'Final answer:' found -> PARSING WILL WORK")
        else:
            print(f"  'Final answer:' NOT found -> PARSING WILL FAIL (returns None)")
    except Exception as e:
        print(f"  FAILED: {e}")

    print()
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"  Working models: {working_models}")
    print(f"  Suggested replacement: {working_models[0]}")
    print(f"  Update gen_conf['gemini-1-5']['model'] in debate_teachers_student_w_critique.py")
else:
    print()
    print("NO WORKING MODELS FOUND. Check your GEMINI_API_KEY.")

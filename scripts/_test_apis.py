"""Quick test: verify all API keys work."""
import litellm
import os
from dotenv import load_dotenv
load_dotenv()

models = {
    "gemini/gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite (Google)",
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash (Google)",
    "groq/llama-3.3-70b-versatile": "Llama 3.3 70B (Groq FREE)",
    "groq/qwen-qwq-32b": "Qwen QwQ 32B (Groq FREE)",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout (Groq FREE)",
    "gpt-4o-mini": "GPT-4o-mini (OpenAI)",
    "gpt-4o": "GPT-4o (OpenAI)",
}

working = []
for model_id, name in models.items():
    print(f"Testing {name}...")
    try:
        r = litellm.completion(
            model=model_id,
            messages=[{"role": "user", "content": "Reply ONLY with this JSON: {\"status\": \"ok\"}"}],
            max_tokens=30,
        )
        content = r.choices[0].message.content or ""
        txt = content.strip()[:80]
        tok_in = r.usage.prompt_tokens if r.usage else 0
        tok_out = r.usage.completion_tokens if r.usage else 0
        print(f"  OK: {txt}  [tokens: {tok_in}in/{tok_out}out]")
        working.append(model_id)
    except Exception as e:
        print(f"  FAILED: {str(e)[:200]}")

print(f"\n{len(working)}/{len(models)} models working: {working}")

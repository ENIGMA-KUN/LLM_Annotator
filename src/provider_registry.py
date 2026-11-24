"""
Multi-provider LLM registry.
Supports cloud APIs (OpenAI, Anthropic, Google, Mistral, Cohere)
and local endpoints (Ollama, LM Studio, vLLM, any OpenAI-compatible).
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path(__file__).resolve().parent.parent / ".llm_providers.json"

CLOUD_PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "key_prefix": "sk-",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3", "o4-mini"],
    },
    "anthropic": {
        "label": "Anthropic",
        "base_url": "https://api.anthropic.com",
        "key_prefix": "sk-ant-",
        "models": ["claude-sonnet-4-5-20250514", "claude-haiku-4-5-20251001", "claude-opus-4-20250514"],
    },
    "google": {
        "label": "Google (Gemini)",
        "base_url": None,
        "key_prefix": "AI",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    },
    "mistral": {
        "label": "Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "key_prefix": "",
        "models": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "codestral-latest"],
    },
    "cohere": {
        "label": "Cohere",
        "base_url": "https://api.cohere.com/v2",
        "key_prefix": "",
        "models": ["command-r-plus", "command-r", "command-light"],
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "key_prefix": "sk-",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
}

LOCAL_PROVIDERS = {
    "ollama": {
        "label": "Ollama (Local)",
        "default_url": "http://localhost:11434/v1",
        "requires_key": False,
        "models": ["llama3.1", "llama3.2", "mistral", "gemma2", "phi3", "qwen2.5"],
    },
    "lm_studio": {
        "label": "LM Studio (Local)",
        "default_url": "http://localhost:1234/v1",
        "requires_key": False,
        "models": [],
    },
    "vllm": {
        "label": "vLLM (Local/Remote)",
        "default_url": "http://localhost:8000/v1",
        "requires_key": False,
        "models": [],
    },
    "custom": {
        "label": "Custom OpenAI-Compatible",
        "default_url": "",
        "requires_key": False,
        "models": [],
    },
}


@dataclass
class ProviderConfig:
    provider_id: str
    label: str
    api_key: str = ""
    base_url: str = ""
    selected_model: str = ""
    is_local: bool = False
    enabled: bool = False
    custom_models: list = field(default_factory=list)


class ProviderRegistry:
    """Manages LLM provider configurations. Persists to a local JSON file."""

    def __init__(self):
        self.providers: dict[str, ProviderConfig] = {}
        self._load()

    def _load(self):
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                for pid, cfg in data.items():
                    self.providers[pid] = ProviderConfig(**cfg)
            except (json.JSONDecodeError, TypeError):
                self._init_defaults()
        else:
            self._init_defaults()

    def _init_defaults(self):
        for pid, info in CLOUD_PROVIDERS.items():
            self.providers[pid] = ProviderConfig(
                provider_id=pid,
                label=info["label"],
                base_url=info.get("base_url", ""),
            )
        for pid, info in LOCAL_PROVIDERS.items():
            self.providers[pid] = ProviderConfig(
                provider_id=pid,
                label=info["label"],
                base_url=info.get("default_url", ""),
                is_local=True,
            )

    def save(self):
        data = {pid: asdict(cfg) for pid, cfg in self.providers.items()}
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def set_key(self, provider_id: str, api_key: str):
        if provider_id in self.providers:
            self.providers[provider_id].api_key = api_key
            self.providers[provider_id].enabled = bool(api_key)
            self.save()

    def set_url(self, provider_id: str, base_url: str):
        if provider_id in self.providers:
            self.providers[provider_id].base_url = base_url
            self.save()

    def set_model(self, provider_id: str, model: str):
        if provider_id in self.providers:
            self.providers[provider_id].selected_model = model
            self.save()

    def enable(self, provider_id: str, enabled: bool = True):
        if provider_id in self.providers:
            self.providers[provider_id].enabled = enabled
            self.save()

    def get_active(self) -> Optional[ProviderConfig]:
        for cfg in self.providers.values():
            if cfg.enabled:
                return cfg
        return None

    def get_active_providers(self) -> list[ProviderConfig]:
        return [cfg for cfg in self.providers.values() if cfg.enabled]

    def get_models(self, provider_id: str) -> list[str]:
        if provider_id in CLOUD_PROVIDERS:
            base = CLOUD_PROVIDERS[provider_id]["models"]
        elif provider_id in LOCAL_PROVIDERS:
            base = LOCAL_PROVIDERS[provider_id]["models"]
        else:
            base = []
        custom = self.providers.get(provider_id, ProviderConfig(provider_id, "")).custom_models
        return base + [m for m in custom if m not in base]

    def add_custom_model(self, provider_id: str, model_name: str):
        if provider_id in self.providers and model_name:
            if model_name not in self.providers[provider_id].custom_models:
                self.providers[provider_id].custom_models.append(model_name)
                self.save()

    def test_connection(self, provider_id: str) -> tuple[bool, str]:
        cfg = self.providers.get(provider_id)
        if not cfg:
            return False, "Provider not found"

        try:
            if provider_id == "google":
                import google.generativeai as genai
                genai.configure(api_key=cfg.api_key)
                models = genai.list_models()
                return True, f"Connected: {sum(1 for _ in models)} models available"

            elif provider_id == "anthropic":
                from anthropic import Anthropic
                client = Anthropic(api_key=cfg.api_key)
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "hi"}],
                )
                return True, "Connected"

            elif provider_id in ("openai", "deepseek", "mistral") or cfg.is_local:
                from openai import OpenAI
                client = OpenAI(
                    api_key=cfg.api_key or "not-needed",
                    base_url=cfg.base_url,
                )
                models = client.models.list()
                names = [m.id for m in models.data[:5]]
                return True, f"Connected: models: {', '.join(names)}"

            elif provider_id == "cohere":
                import cohere
                client = cohere.ClientV2(api_key=cfg.api_key)
                return True, "Connected"

            else:
                return False, "Unknown provider type"

        except Exception as e:
            return False, str(e)[:200]

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

DEFAULT_CHAT_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
DEFAULT_EMBEDDING_KEY = os.environ.get("EMBEDDING_KEY", "text-embedding-3-small")
# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


CHAT_MODELS = {
    "openai": "gpt-5-mini",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash-lite",
    "ollama": "dolphin-llama3:latest",
}

CHAT_PRICING: dict[tuple[str, str], dict[str, float]] = {
    ("openai", "gpt-4o-mini"): {"input": 0.15, "output": 0.60},
    ("groq", "llama-3.3-70b-versatile"): {"input": 0.59, "output": 0.79},
    ("gemini", "gemini-2.5-flash-lite"): {"input": 0.10, "output": 0.40},
}


def estimate_cost(
        provider: str | None,
        model: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
) -> float | None:
    """
    Rough USD cost estimate from CHAT_PRICING. Returns None if the
    provider/model isn't in the table (e.g. Ollama, or a model picked via
    live model-discovery that we don't have prices for) or token counts
    are missing.
    """
    if not provider or not model or input_tokens is None or output_tokens is None:
        return None
    prices = CHAT_PRICING.get((provider, model))
    if not prices:
        return None
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def get_chat_model(
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        streaming: bool = False,
):
    """
    Returns LangChain Chat-Model.

    Provider: openai | groq | gemini | ollama
    """

    provider = provider or DEFAULT_CHAT_PROVIDER
    model = model or CHAT_MODELS.get(provider)

    kwargs = {
        "model": model,
        "temperature": temperature,
        "streaming": streaming,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if provider == "openai":
        return ChatOpenAI(**kwargs)
    elif provider == "groq":
        return ChatGroq(**kwargs)
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(**kwargs)
    elif provider == "ollama":
        return ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        raise ValueError(
            f"Unknown chat provider '{provider}'. Available: openai, groq, gemini, ollama"
        )


def resolve_chat_config(provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    """
    Resolves chat config or uses default
    """
    resolved_provider = provider or DEFAULT_CHAT_PROVIDER
    resolved_model = model or CHAT_MODELS.get(resolved_provider)
    return resolved_provider, resolved_model


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


EMBEDDING_CONFIGS: dict[str, dict] = {
    "nomic-embed-text": {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "dim": 768,
        "table": "media_chunks_nomic_embed_text",
    },
    "mxbai-embed-large": {
        "provider": "ollama",
        "model": "mxbai-embed-large",
        "dim": 1024,
        "table": "media_chunks_mxbai_embed_large",
    },
    "text-embedding-3-small": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dim": 1536,
        "table": "media_chunks_text_embedding_3_small",
    },
    "gemini-embedding-001-768": {
        "provider": "google",
        "model": "gemini-embedding-001",
        "dim": 768,
        "table": "media_chunks_gemini_embedding_001_768",
        "extra_kwargs": {"output_dimensionality": 768},
    },
    "gemini-embedding-001-3072": {
        "provider": "google",
        "model": "gemini-embedding-001",
        "dim": 3072,
        "table": "media_chunks_gemini_embedding_001_3072",
        "extra_kwargs": {},  # 3072 ist die native Ausgabegröße, keine Truncation nötig
    },
    "full-text": {
        "provider": "none",  # kein echtes Embedding-Modell, siehe rag_service.py
        "model": None,
        "dim": None,
        "table": None,
    },
}


def get_embedding_model(key: str | None = None):
    """
    Returns LangChain Embeddings object for the given embedding-config key.
    """
    key = key or DEFAULT_EMBEDDING_KEY
    cfg = EMBEDDING_CONFIGS.get(key)
    if cfg is None:
        raise ValueError(
            f"Unknown embedding key '{key}'. Available: {', '.join(EMBEDDING_CONFIGS)}"
        )

    provider = cfg["provider"]
    model = cfg["model"]
    extra = cfg.get("extra_kwargs", {})

    if provider == "ollama":
        return OllamaEmbeddings(
            model=model,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    elif provider == "openai":
        return OpenAIEmbeddings(model=model, **extra)
    elif provider == "google":
        return GoogleGenerativeAIEmbeddings(model=model, **extra)
    else:
        raise ValueError(f"Unknown embedding provider '{provider}' for key '{key}'")


def get_embedding_dim(key: str | None = None) -> int:
    """Returns the vector dimension for the given provider."""
    key = key or DEFAULT_EMBEDDING_KEY
    return EMBEDDING_CONFIGS[key]["dim"]


def get_embedding_table(key: str | None = None) -> str:
    key = key or DEFAULT_EMBEDDING_KEY
    return EMBEDDING_CONFIGS[key]["table"]


def resolve_embedding_key(key: str | None = None) -> str:
    """Resolves embedding key or falls back to default."""
    return key or DEFAULT_EMBEDDING_KEY


def build_chunk_model_registry() -> dict[str, type]:
    """
    Maps key to embedding table.
    """
    from models import Base  # lokal, um zirkuläre Imports zu vermeiden

    table_to_model = {
        mapper.class_.__tablename__: mapper.class_
        for mapper in Base.registry.mappers
    }

    registry: dict[str, type] = {}
    for key, cfg in EMBEDDING_CONFIGS.items():
        if cfg.get("table") is None:
            registry[key] = None
            continue
        model_cls = table_to_model.get(cfg["table"])
        if model_cls is None:
            raise RuntimeError(
                f"Kein SQLAlchemy-Model mit __tablename__ = '{cfg['table']}' "
                f"gefunden (für Embedding-Key '{key}'). Model vergessen?"
            )
        registry[key] = model_cls
    return registry

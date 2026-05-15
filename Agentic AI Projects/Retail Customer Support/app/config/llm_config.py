from app.config.settings import get_settings

settings = get_settings()


def get_llm_config() -> dict:
    return {
        "model": settings.openai_model,
        "temperature": settings.llm_temperature,
        "api_key": settings.openai_api_key,
        "enabled": bool(settings.openai_api_key),
    }

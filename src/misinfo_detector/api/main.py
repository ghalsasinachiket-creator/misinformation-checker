from fastapi import FastAPI

from misinfo_detector.config import get_settings, load_training_config


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Detects misleading image-caption pairs with CLIP-based multimodal inference.",
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "environment": settings.app_env}

    @app.get("/ready", tags=["system"])
    def ready() -> dict[str, str]:
        config = load_training_config(settings.config_path)
        model_name = config.get("model", {}).get("clip_model_name", "unknown")
        return {"status": "ready", "clip_model": model_name}

    return app


app = create_app()

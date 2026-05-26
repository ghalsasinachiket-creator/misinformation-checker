# misinformation-checker

A multimodal misinformation detector that flags misleading image-caption pairs using a fine-tuned CLIP model with GradCAM explainability. Built with PyTorch, HuggingFace Transformers, and the NewsCLIPpings dataset.

## Current Milestone

Step 1 establishes a deployable service skeleton:

- FastAPI application with `/health` and `/ready` endpoints
- Environment-driven settings
- Training configuration scaffold for CLIP fine-tuning
- Dockerfile and Make targets for local development
- Tests for the service readiness surface

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
make install
make test
make run
```

The API will run at `http://localhost:8000`.

## Docker

```bash
docker build -t misinformation-checker .
docker run --rm -p 8000:8000 misinformation-checker
```

.PHONY: install test lint run

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

run:
	PYTHONPATH=src $(PYTHON) -m uvicorn misinfo_detector.api.main:app --host 0.0.0.0 --port 8000 --reload

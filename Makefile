PYTHON ?= python

.PHONY: install install-local install-server demo test api benchmark-models

install:
	$(PYTHON) -m pip install -r requirements.txt

install-local:
	$(PYTHON) -m pip install -r requirements-local-llm.txt

install-server:
	$(PYTHON) -m pip install -r requirements-server.txt

demo:
	$(PYTHON) scripts/run_demo.py

test:
	$(PYTHON) -m pytest -q

api:
	$(PYTHON) -m uvicorn app.api:app --host 0.0.0.0 --port 8000

benchmark-models:
	$(PYTHON) scripts/benchmark_local_models.py

PYTHON ?= python

.PHONY: install demo test api lint

install:
	$(PYTHON) -m pip install -r requirements.txt

demo:
	$(PYTHON) scripts/run_demo.py

test:
	$(PYTHON) -m pytest -q

api:
	$(PYTHON) -m uvicorn app.api:app --host 0.0.0.0 --port 8000


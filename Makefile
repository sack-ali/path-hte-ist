.PHONY: install setup data clean preprocess test smoke dash lint

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

setup: install
	@echo "Now download IST_corrected.csv from https://datashare.ed.ac.uk/handle/10283/124"
	@echo "and place it at data/raw/IST_corrected.csv"

preprocess:
	python -m src.preprocess

smoke:
	python -m tests.test_pipeline_synth

test:
	pytest tests/ -v

dash:
	streamlit run dashboard/app.py

lint:
	ruff check src/ dashboard/ tests/

clean:
	rm -rf data/processed/* reports/figures/*

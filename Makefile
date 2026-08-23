.PHONY: setup test examples report serve clean

PYTHON := .venv/bin/python

setup:
	./scripts/setup.sh

test:
	$(PYTHON) -m pytest --cov=hexaform --cov-report=term-missing

examples:
	$(PYTHON) scripts/run_examples.py

report:
	$(PYTHON) scripts/generate_technical_report.py

serve:
	.venv/bin/hexaform serve

clean:
	@echo "Generated artifacts are under output/. Remove selected runs there when no longer needed."

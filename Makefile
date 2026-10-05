# Convenience targets. Every target is a thin wrapper around a documented command.
PY ?= python
LATEX ?= pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: help setup test fast all report slides clean app-setup app-train app docker

help:
	@echo "make setup   - create .venv and install pinned requirements"
	@echo "make test    - run the unit tests (no dataset needed)"
	@echo "make fast    - quick smoke run of the full pipeline (subsample)"
	@echo "make all     - reproduce every number and figure (python -m src.run_all)"
	@echo "make report  - build report/report.pdf (run 'make all' first)"
	@echo "make slides  - build presentation/slides.pdf (run 'make all' first)"
	@echo "make app-train / app / docker - train and serve the web interface"

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

test:
	$(PY) -m pytest -q

fast:
	$(PY) -m src.run_all --fast

all:
	$(PY) -m src.run_all

report:
	cd report && $(LATEX) report.tex >/dev/null && $(LATEX) report.tex | tail -2

slides:
	cd presentation && $(LATEX) slides.tex >/dev/null && $(LATEX) slides.tex | tail -2

clean:
	rm -rf results report/generated report/figures/*.pdf
	cd report && rm -f *.aux *.log *.out
	cd presentation && rm -f *.aux *.log *.out *.nav *.snm *.toc *.vrb

app-setup:
	.venv/bin/pip install -r requirements-app.txt

app-train:
	$(PY) -m app.train

app:
	$(PY) -m uvicorn app.main:app --port 8000

docker:
	docker build -t binaaz-estimator . && docker run --rm -p 8000:8000 -v "$(CURDIR)/models:/app/models:ro" binaaz-estimator

.PHONY: install run test lint

install:
	pip install --upgrade pip
	pip install -r requirements.txt

run:
	python final_with_markdown.py

test:
	pytest

lint:
	flake8 .

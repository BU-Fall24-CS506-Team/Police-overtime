# Makefile for managing the Python project

.PHONY: install run

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Run the Jupyter Notebook
run:
	jupyter nbconvert --to notebook --execute data_cleaning.ipynb --output executed_data_cleaning.ipynb

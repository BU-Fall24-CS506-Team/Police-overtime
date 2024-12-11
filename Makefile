.PHONY: install run clean venv

# Define variables
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
JUPYTER = $(VENV_DIR)/bin/jupyter

# Create virtual environment
venv:
	python3 -m venv $(VENV_DIR)

# Install dependencies inside the virtual environment
install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Run the Jupyter Notebook in the virtual environment
run: install
	$(JUPYTER) nbconvert --to notebook --execute final.ipynb --output final.ipynb

# Clean up generated files (optional)
clean:
	rm -rf $(VENV_DIR)

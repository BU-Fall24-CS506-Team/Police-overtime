.PHONY: install run clean venv

# Define variables
VENV_DIR = .venv
ACTIVATE = . $(VENV_DIR)/bin/activate
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
JUPYTER = $(VENV_DIR)/bin/jupyter

# Create virtual environment
venv:
	python3 -m venv $(VENV_DIR)

# Install dependencies inside the virtual environment
install: venv
	$(ACTIVATE) && $(PIP) install --upgrade pip
	$(ACTIVATE) && $(PIP) install -r requirements.txt

# Run the Jupyter Notebook in the virtual environment
run: install
	$(ACTIVATE) && $(JUPYTER) nbconvert --to notebook --execute final.ipynb --output final.ipynb

# Clean up generated files (optional)
clean:
	rm -rf $(VENV_DIR)

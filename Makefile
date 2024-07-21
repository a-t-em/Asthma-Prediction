# Run setup and launch app
all: setup launch_app

# Set up the virtual environment
setup:
	python -m venv venv
	pip install -r requirements.txt

# Run the Flask app
launch_app:
	python scoring_script.py

# Run tests only after launching the app
test:
	pytest testing.py

lint:
	pylint build_model.py scoring_script.py testing.py

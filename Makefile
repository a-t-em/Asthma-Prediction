# Run setup and launch app
all: setup build_model launch_app

# Set up the virtual environment
setup:
	python -m venv venv
	pip install -r requirements.txt

# Train model and deploy as web service (Flask app) 
run: build_model launch_app

# Launch mlflow server with SQL backend and run script for training model
build_model:
	mlflow server --backend-store-uri sqlite:///backend.db &
	python build_model.py

# Run the Flask app
launch_app:
	python scoring_script.py

# Run tests only after launching the app
test: 
	pytest testing.py

lint:
	pylint build_model.py scoring_script.py testing.py

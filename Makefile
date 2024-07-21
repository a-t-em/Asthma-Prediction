# Run setup and all scripts
all: setup launch_app

# Set up the virtual environment
setup:
	python -m venv venv
	pip install -r requirements.txt

# Run the scoring script (Flask app)
launch_app:
	python scoring_script.py

# Run tests after launching the app
test:
	python scoring_script.py & 
	sleep 5 
	pytest testing.py

lint:
	pylint build_model.py scoring_script.py testing.py

# Build the model and launch MLflow UI
build_model:
	mlflow ui 
	python build_model.py
	kill $(lsof -t -i:5000)

# Run Evidently metrics notebook and Evidently UI
evidently_metrics:
	python evidently-metrics.ipynb
	evidently ui
	killall evidently-ui

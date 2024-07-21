This is a submission of the final project for [DataTalksClub's MLOps Zoomcamp 2024](https://github.com/DataTalksClub/mlops-zoomcamp). The purpose of the project is to demonstrate the ability to create a machine learning pipeline for a self-selected problem that adheres to MLOps best practices as taught during the course (e.g. with reusable code functions, monitoring of training metrics, testing, accessible prediction endpoint etc.)

## **Background**

[Asthma](https://www.who.int/news-room/fact-sheets/detail/asthma) is a chronic lung disease that affects the airways, causing them to narrow and swell. Common symptoms include coughing, wheezing, chest tightness, and shortness of breath. 

Machine learning methods can help identify early signs of asthma in patients by analyzing features like wheezing and exposure to various environmental allergens and irritants.

## **Dataset - `asthma_disease_data.csv`**

Source: Rabie El Kharoua. (2024). 🌬️ Asthma Disease Dataset 🌬️ [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/8669080 

Brief Description: This is a small, curated dataset containing anonymized health data of 2,392 patients, a minority of which have been diagnosed with Asthma Disease. 

## **Model - `model.pkl`**

The model is a simple Random Forest binary classification model that has been fine-tuned using **hyperopt** over a limited range of hyperparameters. 

Since it is a decision-tree based model and the original dataset is well-curated with no missing values, no preprocessing of data is necessary. However, since the dataset is highly imbalanced, with asthma cases being a minority, minority oversampling techniques were used to create a balanced dataset for training.

Model inputs: select features with relatively higher correlation with the target.

Model outputs: an array containing the value 0 for a negative diagnosis (no asthma) and the value 1 for a positive diagnosis (asthma).

## **Files**

`exploration.ipynb` -- a notebook initially published on Kaggle that conducts exploratory data analysis on the dataset and examines the efficacy of the default Random Forest Classifier. The F1 score of 0.93 yielded by the model trained on a handful of select features on the withheld 20% of the dataset suggests that a decision tree based model is well-suited for the problem.

`build_model.py` -- a script for loading and preprocessing the dataset from the CSV file and using hyperopt to fine-tune a Random Forest Classifier on it. The best performing model from 10 hyperopt trials is saved as `model.pkl`, and it is used to make predictions on the test dataset. The predictions are then saved together with the corresponding features in a CSV file under the `data` folder.

`scoring_script.py` -- a script for launching a Flask app (web service) locally that allows for posting a data record input in JSON format to a prediction endpoint that scores the input using `model.pkl`. The app returns a prediction between 0 and 1 (inclusive) within a JSON object.

`testing.py` -- a script for testing the Flask app from `scoring_script.py` to ensure that it returns a prediction between 0 and 1 (inclusive). Enter `pytest testing.py` into the terminal to run it.

`evidently-metrics.ipynb` -- a notebook creating monitoring reports and dashboard comparing the training dataset and the test dataset (with predictions included) using [Evidently](https://www.evidentlyai.com/). To view the dashboard in the browser locally, enter `evidently ui` in the terminal after running the entire notebook to create the workspace and the associated dashboard panels and reports. 

## **Usage**

Clone the repository and run `make setup` in the terminal (which requires [Make](https://www.technewstoday.com/install-and-use-make-in-windows/) to be installed first). This will launch a virtual environment and install all dependencies from `requirements.txt`. You can then launch the Flask app with `make launch_app` and then test it using `make test` command. 

Alternatively, you can run `make all` to set up the environment and launch the Flask app.

## **Dependencies - `requirements.txt`**
 
To replicate the development environment, clone the respository onto your local machine and run the command `pip install -r requirements.txt` in your terminal. 

The main libaries utilized are listed below:

io<br>
os<br>
pickle<br>
json<br>
requests<br>
pandas<br>
numpy<br>
sklearn<br>
imbalanced-learn<br>
seaborn<br>
matplotlib<br>
mlflow<br>
pytest<br>
hyperopt<br>
flask<br>
evidently<br>
pylint<br>



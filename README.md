This is a submission for [DataTalksClub's MLOps Zoomcamp 2024](https://github.com/DataTalksClub/mlops-zoomcamp).

**Dataset**

Source: Rabie El Kharoua. (2024). 🌬️ Asthma Disease Dataset 🌬️ [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/8669080 

Brief Description: This is a small, curated dataset containing anonymized health data of 2,392 patients, a minority of which have been diagnosed with Asthma Disease. 

**Model**

The model is a simple Random Forest binary classification model. Since it is a decision-tree based model and the original dataset is well-curated with no missing values, no preprocessing of data is necessary. However, since the dataset is highly imbalanced, with asthma cases being a minority, minority oversampling techniques were used to create a balanced dataset for training.

Model inputs: select features with relatively higher correlation with the target.

Model outputs: an array containing the value 0 for a negative diagnosis (no asthma) and the value 1 for a positive diagnosis (asthma)

**Dependencies**
pandas
numpy
sklearn
imbalanced-learn
seaborn
matplotlib
mlflow
pytest
pickle
io
json




# Machine Learning Project Pipeline

This project demonstrates a complete machine learning pipeline, including data processing, model training, evaluation, and inference, followed by visualizations using Streamlit dashboards.

## Setup Instructions

### 1. Clone the Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/minimac321/Attrition-model.git
cd Attrition-model
```


### 2. Set Up a Virtual Environment
To keep the project dependencies isolated, it's recommended to create a virtual environment:

On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```


### 3. Install Required Libraries
Once the virtual environment is activated, install the required libraries:


```bash
pip install -r requirements.txt
```
This will install all necessary Python libraries for the project.



### 4. Start MLflow Tracking Server
This project utilizes MLflow for tracking experiments. To start the MLflow server locally with its UI interface, run the following command:

```bash
mlflow ui
```
This will start the MLflow UI at `http://localhost:8501`, where you can track your experiment runs, model parameters, metrics, and artifacts.





##  Workflow Steps
### Step 1: Data Processing
The first step is to process the raw data and split it into training, validation, and test datasets. Use the data_processing.py script for this purpose. The script reads a raw .xlsx file and processes the data into the required format.

Run the following command to generate the processed datasets:

```bash
python scripts/data_processing.py
```
This will create the train, valid, and test datasets in the appropriate directories under data/.

### Step 2: Model Training
Once the data is processed, you can train the machine learning model by running the model_training.py script. This script will train a model on the training dataset, log parameters, metrics, and artifacts (including the trained model) to MLflow.

Run the following command to train the model:

```bash
python scripts/model_training.py
```
After training is completed, go to the MLflow UI (http://localhost:5000) to get the URI of the best model logged by the training script.

### Step 3: Model Evaluation
After obtaining the model URI from MLflow, run the model_evaluation.py script to evaluate the model on the test dataset. You need to pass the model URI (from the previous step) to this script.

Run the following command to evaluate the model:

```bash
python scripts/model_evaluation.py  <mlflow-model-uri>
```
Example: 
```bash
python scripts/model_evaluation.py runs:/1ca282ff0e1f4442981544ab6303952a/model
```
This script will evaluate the model's performance and log the evaluation metrics and plots to MLflow.

### Step 4: Model Inference
Once the model has been trained and evaluated, you can run the inference on new or unseen data using the model_inference.py script. Like the evaluation script, this also requires the model URI from MLflow.

Run the following command for model inference:

```bash
python scripts/model_inference.py <mlflow-model-uri>
```
Example: 
```bash
python scripts/model_inference.py runs:/1ca282ff0e1f4442981544ab6303952a/model
```


## Streamlit Dashboards
After running the training, evaluation, and inference steps, you can visualize the data and results using two Streamlit applications.

### Step 5: Start Streamlit Dashboards
EDA Dashboard: This dashboard provides Exploratory Data Analysis (EDA) on the data.

Run the following command to start the EDA dashboard:
```bash
streamlit run streamlit_dashboards/dashboard_eda.py
```
This will launch a Streamlit app that displays insights and visualizations of the dataset.

Evaluation Dashboard: This dashboard visualizes the model evaluation results fetched from MLflow.

Run the following command to start the evaluation dashboard:
```bash
streamlit run streamlit_dashboards/dashboard_evaluation.py
```
This will launch another Streamlit app that shows the evaluation metrics and plots, such as confusion matrices, feature importance, and more.



## Summary of the Project Pipeline
1. Data Processing: Use `data_processing.py` to prepare datasets.
2. Model Training: Run `model_training.py` to train the model and log it to MLflow.
3. Model Evaluation: Use `model_evaluation.py` to evaluate the model and log the results to MLflow.
4. Model Inference: Run `model_inference.py` to generate predictions on new data.
5. Streamlit Dashboards: Launch `dashboard_eda.py` for data analysis and `dashboard_evaluation.py` for evaluation results.
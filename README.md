# AIT-511 Machine Learning Project 2

**Team Members:**
* Pratheek P (MS2025010)
* Abu Talha (MT2025703)

## Project Overview
This project implements and evaluates machine learning models for two distinct classification challenges:

1.  **Smoker Status Prediction (Binary Classification):**
    * **Goal:** Predict whether a patient is a smoker based on bio-signals (blood pressure, cholesterol, etc.).
    * **Challenge:** Detecting positive cases (Recall) in a health context.
    * **Winner:** Neural Network.

2.  **Forest Cover Type (Multiclass Classification):**
    * **Goal:** Classify forest areas into 7 distinct tree types (Spruce, Pine, Aspen, etc.) using cartographic variables.
    * **Challenge:** Severe class imbalance (Class 1 & 2 dominate) and non-linear feature interactions.
    * **Winner:** Neural Network.

## Repository Structure
* `notebooks/`:
    * `Smoker Status Prediction.ipynb`: Complete analysis for the Smoker dataset.
    * `Forest_Cover_Type.ipynb`: Complete analysis for the Forest Cover dataset.
* `images/`: Visualizations generated during EDA and model evaluation.
* `README.md`: Project documentation.

## How to Run (Google Colab)
This project was designed for **Google Colab**. Follow these steps to reproduce our results:

### 1. Download Data
Get the datasets from Kaggle:
* **Smoker Dataset:** [Download Here](https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals) -> Save as `train_dataset.csv`
* **Forest Dataset:** [Download Here](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset) -> Save as `covtype.csv`

### 2. Launch Notebooks
Upload the `.ipynb` files from the `notebooks/` folder to [Google Colab](https://colab.research.google.com/).

### 3. Upload Data to Runtime
**Critical Step:** The notebooks read files from the local environment.
1.  In Colab, open the **Files** sidebar.
2.  Drag and drop `train_dataset.csv` (for Smoker notebook) or `covtype.csv` (for Forest notebook).
3.  Wait for the upload to complete.

### 4. Execute
Go to **Runtime** > **Run all**.
The notebook will automatically perform data cleaning, feature engineering, hyperparameter tuning, and final evaluation.

## Methodology
* **Preprocessing:** Log-transformations for skewed features, circular encoding for aspect angles, and standard scaling.
* **Tuning Strategy:** `RandomizedSearchCV` on a stratified subset (10k-25k samples) to efficiently find optimal hyperparameters.
* **Models:** Logistic Regression (Baseline), Support Vector Machines (SVM), and Neural Networks (MLP).
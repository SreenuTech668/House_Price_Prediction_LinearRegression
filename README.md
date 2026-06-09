# House_Price_Prediction_LinearRegression
# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts house prices using Machine Learning and Linear Regression. The application is built with Streamlit and allows users to upload a housing dataset, train a model, and view prediction results along with the R² score.

The project demonstrates the complete Machine Learning workflow, including:

* Data Loading
* Data Preprocessing
* Missing Value Handling
* One-Hot Encoding
* Train-Test Splitting
* Model Training
* Model Evaluation
* Streamlit Deployment

---

## 🚀 Features

* Upload CSV datasets directly from the web interface
* Automatic handling of missing values
* Automatic encoding of categorical features
* Linear Regression model training
* R² Score calculation
* Actual vs Predicted price comparison
* Dataset statistics and missing value report

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit

---

## 📂 Project Structure

```text
House_Price_Prediction/
│
├── app.py
├── train.csv
├── requirements.txt
└── README.md
```

---

## 📊 Machine Learning Workflow

### 1. Load Dataset

The dataset is uploaded through the Streamlit interface and loaded using Pandas.

### 2. Data Preprocessing

* Numerical missing values are replaced using the median.
* Categorical missing values are replaced using the most frequent value.
* Categorical features are converted into numerical features using One-Hot Encoding.

### 3. Train-Test Split

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

### 4. Model Training

A Linear Regression model is trained using the processed dataset.

### 5. Model Evaluation

The model performance is measured using the R² Score.

R² Score Formula:

R² = 1 - (Σ(y - ŷ)² / Σ(y - ȳ)²)

Where:

* y = Actual Value
* ŷ = Predicted Value
* ȳ = Mean of Actual Values

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/House_Price_Prediction.git
```

Navigate to the project directory:

```bash
cd House_Price_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

After running the command, Streamlit will open the application in your browser.

---

## 📈 Sample Output

The application displays:

* Dataset Preview
* R² Score
* Training Score
* Actual vs Predicted Prices
* Dataset Information
* Missing Values Summary

---

## 🎯 Learning Outcomes

Through this project, I learned:

* Data preprocessing techniques
* Feature engineering concepts
* Handling categorical variables
* Building machine learning pipelines
* Evaluating regression models
* Deploying ML applications using Streamlit

---

## 🔮 Future Improvements

* Random Forest Regressor
* XGBoost Regressor
* Hyperparameter Tuning
* Model Comparison Dashboard
* House Price Prediction for User Inputs
* Deployment on Streamlit Cloud

---

## 👨‍💻 Author

Sreenu Tech

Aspiring Machine Learning Engineer passionate about Python, SQL, Machine Learning, Deep Learning, LLMs, RAG, LangChain, LangGraph, and AI-powered applications.

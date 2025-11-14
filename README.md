# 🏠 Boston House Price Prediction App

This is a machine learning project that predicts house prices using the classic Boston Housing dataset. The project includes data analysis, model training (specifically addressing overfitting with regularization), and a final interactive web application built with Streamlit.

## 🚀 Features

* **Interactive Web UI:** A simple and clean user interface built with Streamlit.
* **Real-Time Predictions:** Enter 13 house features (like crime rate, number of rooms, etc.) and get an instant price prediction.
* **Regularized Model:** Uses a **Ridge (L2) Regression** model, which was chosen over a standard Linear Regression to prevent overfitting and ensure more reliable predictions on new data.

## 📸 App Demo
![My App Demo](demo.png)
![My App Output](output.png)

## 💡 Model Details

A key part of this project was comparing a standard `LinearRegression` model with regularized models.

* The initial model showed signs of **severe overfitting**, with a near-perfect score on the training data but a much lower, scattered score on the test data.
* By implementing **Ridge (L2)** and **Lasso (L1)** regularization, we created a model that generalizes much better. The final app uses the Ridge model, which provides a more realistic and stable performance on unseen data.




## 🛠️ Technologies Used

* **Python:** Core programming language
* **Pandas:** For data loading and manipulation
* **Scikit-learn:** For model training (`Ridge`), preprocessing (`StandardScaler`), and evaluation
* **Joblib:** For saving and loading the trained model and scaler
* **Streamlit:** For building and serving the interactive web app

## 🏃 How to Run This Project

Follow these steps to get the app running on your local machine.

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name# House-Price-Prediction

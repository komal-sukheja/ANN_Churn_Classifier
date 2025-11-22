# 📉 Customer Churn Prediction App

A complete end-to-end machine learning project that predicts **customer churn** using an **Artificial Neural Network (ANN)**.  
The model is trained using TensorFlow/Keras and deployed as an interactive web application using **Streamlit**.

---

## 🚀 Project Overview
This project aims to predict whether a customer is likely to churn based on demographic and financial attributes such as age, credit score, balance, salary, activity status, and more.

The solution includes:
- Full preprocessing and feature engineering  
- ANN model training  
- Saving transformers + model  
- A deployed Streamlit web app for real-time predictions  

---

## 🧠 Model Details
**Architecture**
- Dense(64, relu)  
- Dense(32, relu)  
- Dense(1, sigmoid)

**Training Specs**
- Loss: Binary Crossentropy  
- Optimizer: Adam  
- Train/Test Split: 80/20  
- Preprocessing:
  - Label Encoding → Gender  
  - One-Hot Encoding → Geography  
  - Standard Scaling → Numerical features  


---

## 🌐 Streamlit App (UI Features)
- User-friendly form for entering all customer details  
- Real-time churn probability output  
- Color-coded result (Likely / Not Likely to Churn)  
- Automatically loads saved model + encoders + scaler  

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone <your-repo-link>
cd <project-folder>

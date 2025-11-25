# 📉 Customer Churn Prediction App (ANN)

An end-to-end machine learning project that predicts **customer churn** using an **Artificial Neural Network (ANN)**.  
The model is built with TensorFlow/Keras and deployed using **Streamlit** for real-time predictions.

🔗 **Live Streamlit App:** https://annchurnclassifier-hszpd53e63wf.streamlit.app/  
🔗 **GitHub Repository:** https://github.com/komal-sukheja/ANN_Churn_Classifier  

---

##  Project Overview

Customer churn is a major challenge for subscription-based and service-based businesses.  
This project predicts whether a customer is likely to churn based on demographic and financial attributes such as:

- Age  
- Credit Score  
- Gender  
- Geography  
- Balance  
- Salary  
- Tenure  
- Number of Products  
- Activity Status  
- Credit Card Status  

The pipeline includes:
- Data preprocessing & feature engineering  
- Building an ANN classifier  
- Saving model & preprocessing objects  
- Deploying a user-friendly Streamlit web application  

---

##  Model Architecture

The ANN was designed and trained using TensorFlow/Keras.

**Layers**
- Dense(64, activation='relu')  
- Dense(32, activation='relu')  
- Dense(1, activation='sigmoid')  

**Training**
- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Train/Test Split:** 80/20  
- **Metrics:** Accuracy  

**Preprocessing**
- Label Encoding → Gender  
- One-Hot Encoding → Geography  
- Standard Scaling → Numerical columns  

Saved artifacts:
- `model.h5`  
- `label_encoder_gender.pkl`  
- `OHE_geography.pkl`  
- `scaler.pkl`  

---

##  Tech Stack

**Languages & Frameworks**
- Python  
- TensorFlow / Keras  

**Libraries**
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- pickle  

**Deployment**
- Streamlit  

---

## 📈 Output (Model Prediction)

The model returns a churn probability between **0 and 1**.

Prediction logic:
- **> 0.5** → ❌ Customer is **likely to churn**  
- **≤ 0.5** → ✅ Customer is **not likely to churn**

--- 

## 🌐 Streamlit Web App

The deployed app provides:

-  Form-based input for all customer attributes  
-  Real-time churn prediction  
-  Probability score  
-  Color-coded results (Likely / Not Likely to Churn)  
-  Automatic loading of saved model & preprocessing pipeline  

👉 **Try the App:** https://annchurnclassifier-hszpd53e63wf.streamlit.app/

---

## ▶️ How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/komal-sukheja/ANN_Churn_Classifier
cd ANN_Churn_Classifier

# 🏥 Healthcare Data Mining Project

## 👨‍💻 Team Members
- Mohamed Tarek  
- Mohab Mohamed  

---

##  Dataset Source :https://www.kaggle.com/datasets/vanpatangan/readmission-dataset?select=train_df.csv

---

## 📌 Project Overview
This project presents an end-to-end data mining pipeline in the healthcare domain, following the **CRISP-DM methodology**.  
The goal is to analyze patient data, uncover patterns, and generate actionable insights using multiple data mining techniques.

---

## 🎯 Objectives
- Segment patients using clustering techniques  
- Analyze patterns in hospital stay and procedures  
- Investigate readmission behavior  
- Provide business insights through an interactive dashboard  

---

## 📊 Dataset Description
The dataset contains patient-level hospital records including:
- **Age**
- **Gender**
- **Primary Diagnosis**
- **Number of Procedures**
- **Days in Hospital**
- **Comorbidity Score**
- **Discharge Type**
- **Readmission (Target Variable)**

---

## ⚙️ Techniques Used

### 🔹 Clustering
- K-Means clustering for patient segmentation  
- PCA for visualization  
- Silhouette Score for evaluation  

### 🔹 Data Preparation
- Handling missing values  
- One-hot encoding for categorical variables  
- Feature scaling using StandardScaler  

### 🔹 Evaluation
- Cluster profiling (numeric + categorical)  
- Diagnosis distribution across clusters  
- Readmission rate comparison  

---

## 📈 Dashboard Features (Streamlit)
- 📂 Upload dataset dynamically  
- 🎛 Select number of clusters  
- 📊 Cluster size visualization  
- 📉 PCA 2D cluster visualization  
- 🧬 Diagnosis distribution by cluster  
- 🏥 Readmission analysis  
- 📋 Cluster profiling tables  
- 💡 Business insights panel  

---

## 🧠 Key Insights
- Clusters showed **weak separation**, indicating similar patient profiles  
- No strong segmentation based on diagnosis alone  
- Readmission rates are relatively close across clusters  
- Suggests need for richer clinical features for better segmentation  

---

## 🚀 How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run Streamlit app

streamlit run healthcare_clustering_streamlit_app.py

---

## 📂 Project Structure

├── notebook.ipynb
├── healthcare_clustering_streamlit_app.py
├── requirements.txt
├── README.md




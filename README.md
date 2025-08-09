<!-- Project Title Animation -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=900&size=32&pause=1000&color=2E8B57&center=true&vCenter=true&width=1000&lines=AnomVisor;Hyperspectral+Image+Anomaly+Detection;Autoencoder+%2B+Transformer+%2B+SVM" alt="Typing SVG" />
</p>

---

# 🌌 AnomVisor — Hyperspectral Image Anomaly Detection & Classification

**AnomVisor** is an intelligent anomaly detection system designed for **Hyperspectral Image (HSI)** analysis.  
It combines **deep learning (Autoencoder)** with **sequence modeling (Transformer)** and **machine learning (SVM)** to detect and classify spectral anomalies with **>90% accuracy**.

---

## 📖 Domain Overview

### 🌍 What is Hyperspectral Imaging?
Hyperspectral Imaging (HSI) captures images across **hundreds of spectral bands**, far beyond the RGB range.  
It allows us to detect **material properties, chemical compositions, and subtle anomalies** that are invisible to the human eye.

**Applications include:**
- 🚜 **Agriculture** → Crop health monitoring, disease detection.
- 🏭 **Industry** → Quality control, contamination detection.

---

## 🧠 How AnomVisor Works

**Workflow Pipeline:**
1. **Data Acquisition** → Upload `.mat` hyperspectral datasets.
2. **Preprocessing** → PCA dimensionality reduction & patch extraction.
3. **Feature Learning** → Autoencoder learns compact spectral representations.
4. **Sequence Modeling** → Transformer captures contextual relationships.
5. **Anomaly Scoring** → Reconstruction error + Transformer attention.
6. **Classification** → SVM predicts the anomaly class.
7. **Visualization** → Heatmaps & classification labels displayed in the UI.

---

## 📂 Project Structure
```plaintext
AnomVisor/
│── backend/                  # Python backend for ML processing
│   ├── app.py                 # Flask/FastAPI server
│   ├── train_ae_transformer.py# Autoencoder + Transformer training
│   ├── utils.py               # Utility functions
│   ├── requirements.txt       # Python dependencies
│
│── hyperspectral-app/         # React frontend
│   ├── public/                # Public assets
│   ├── src/                   # React components
│   │   ├── components/        # UI Components
│   │   ├── App.js             # Main app entry
│   │   ├── App.css            # Styles
│   ├── package.json           # Frontend dependencies
│
│── uploads/                   # Uploaded datasets
│   ├── Indian_pines_corrected.mat
│
│── AE_Transformer_SVM.ipynb   # Model notebook
│── technical_approach_plan.txt
│── technical_approach_explanation.txt
│── README.md


```
---

 ## 🛠️ Technology Stack

| Language / Tool        | Role |
|------------------------|------|
| **Python**             | Backend ML processing (Flask/FastAPI) |
| **Jupyter Notebook**   | Model prototyping & experiments |
| **JavaScript (React)** | Interactive UI |
| **CSS**                | Styling & layout |
| **HTML**               | UI structure |

<p align="center">
  <img src="https://skillicons.dev/icons?i=css" title="CSS - 28.1%" height="50"/>
  <b>28.1%</b>
  &nbsp;&nbsp;
  <img src="https://skillicons.dev/icons?i=python" title="Python - 24.2%" height="50"/>
  <b>24.2%</b>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/jupyter-notebook/jupyter-notebook.png" title="Jupyter Notebook - 24.0%" height="50"/>
  <b>24.0%</b>
  &nbsp;&nbsp;
  <img src="https://skillicons.dev/icons?i=javascript" title="JavaScript - 23.4%" height="50"/>
  <b>23.4%</b>
  &nbsp;&nbsp;
  <img src="https://skillicons.dev/icons?i=html" title="HTML - 0.3%" height="50"/>
  <b>0.3%</b>
</p>

---

## 📦 Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/AnomVisor.git
cd AnomVisor
```

### 2️⃣ Backend Setup (Python)
```bash

cd backend
pip install -r requirements.txt
python app.py
```
### 3️⃣ Frontend Setup (React)
```bash
cd ../hyperspectral-app
npm install
npm start
```
---
## ▶️ Usage
#### 1.Open the UI in your browser.
#### 2.Upload a .mat dataset (e.g., Indian Pines, Pavia University).
#### 3.Start processing → Real-time progress bar & visual feedback.
#### 4.View results → Anomaly map + classification results.

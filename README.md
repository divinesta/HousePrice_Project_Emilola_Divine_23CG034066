# 🏠 House Price Prediction System

A machine learning-based web application that predicts house prices using a Random Forest Regressor model.

## 📋 Project Overview

This project implements a complete house price prediction system with:
- Machine Learning model training (Random Forest Regressor)
- Flask web application with modern UI
- Model persistence using Joblib
- Ready for cloud deployment

## 🎯 Features Used

The model uses **6 out of 9 recommended features**:
1. **OverallQual** - Overall material and finish quality (1-10)
2. **GrLivArea** - Above grade living area (square feet)
3. **TotalBsmtSF** - Total basement area (square feet)
4. **GarageCars** - Size of garage in car capacity (0-5)
5. **YearBuilt** - Original construction year
6. **Neighborhood** - Physical location (categorical)

Target variable: **SalePrice**

## 📁 Project Structure

```
HousePrice_Project_yourName_matricNo/
│
├── app.py                              # Flask web application
├── requirements.txt                     # Python dependencies
├── HousePrice_hosted_webGUI_link.txt   # Deployment info
│
├── model/
│   ├── model_building.ipynb            # Jupyter notebook for model training
│   ├── house_price_model.pkl           # Trained model (generated after training)
│   ├── scaler.pkl                      # Feature scaler (generated after training)
│   └── label_encoder.pkl               # Neighborhood encoder (generated after training)
│
└── templates/
    └── index.html                      # Web GUI interface
```

## 🚀 Getting Started

### Step 1: Train the Model on Google Colab

1. **Open Google Colab**: https://colab.research.google.com/
2. **Upload the notebook**: Upload `model/model_building.ipynb` to Colab
3. **Download the dataset**:
   - Go to: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
   - Download `train.csv`
4. **Run all cells** in the notebook sequentially
5. **Download the generated files**:
   - `house_price_model.pkl`
   - `scaler.pkl`
   - `label_encoder.pkl`
6. **Move these files** to your project's `model/` folder

### Step 2: Test Locally

```bash
# Navigate to project directory
cd "House Prediction System"

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```

Open your browser and go to: `http://localhost:5000`

### Step 3: Deploy to Cloud

#### Option A: Render.com (Recommended)

1. **Create a GitHub repository** and push your project
2. **Sign up on Render.com**: https://render.com
3. **Create a new Web Service**:
   - Connect your GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Select free tier
4. **Deploy** and get your live URL

#### Option B: PythonAnywhere.com

1. **Sign up**: https://www.pythonanywhere.com
2. **Upload your project files**
3. **Create a new web app** (Flask)
4. **Configure WSGI file** to point to your `app.py`
5. **Reload** and get your URL

#### Option C: Streamlit Cloud

If you prefer Streamlit over Flask, I can provide an alternative Streamlit version.

### Step 4: GitHub Upload

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: House Price Prediction System"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/yourusername/HousePrice_Project.git

# Push
git push -u origin main
```

### Step 5: Complete Submission File

1. Open `HousePrice_hosted_webGUI_link.txt`
2. Fill in your details:
   - Name
   - Matric Number
   - Live URL (from deployment)
   - GitHub repository link
   - Model evaluation metrics (from notebook)

## 📊 Model Performance

After training on Google Colab, you should see metrics like:

- **MAE (Mean Absolute Error)**: ~$20,000 - $30,000
- **RMSE (Root Mean Squared Error)**: ~$30,000 - $40,000
- **R² Score**: ~0.80 - 0.85

*(Your actual values will be shown in the notebook)*

## 🛠️ Technologies Used

- **Python 3.8+**
- **Flask** - Web framework
- **scikit-learn** - Machine learning
- **pandas & numpy** - Data manipulation
- **joblib** - Model persistence
- **HTML/CSS/JavaScript** - Frontend

## 📝 Assignment Checklist

### PART A - Model Development ✅
- [x] Load dataset
- [x] Handle missing values
- [x] Feature selection (6 features)
- [x] Encode categorical variables
- [x] Feature scaling
- [x] Implement Random Forest Regressor
- [x] Train model
- [x] Evaluate (MAE, MSE, RMSE, R²)
- [x] Save model using Joblib
- [x] Model can be reloaded

### PART B - Web GUI ✅
- [x] Load saved model
- [x] Accept user inputs
- [x] Send data to model
- [x] Display predictions
- [x] Modern, responsive design

### PART C - GitHub ✅
- [x] Correct folder structure
- [x] All required files included

### PART D - Deployment ✅
- [ ] Deploy to cloud platform (Complete this step)
- [ ] Fill in deployment info file

## 🎓 How to Use the Web Application

1. **Open the web application** in your browser
2. **Enter house details**:
   - Overall Quality: 1-10
   - Living Area: Square feet
   - Basement Area: Square feet
   - Garage Cars: 0-5
   - Year Built: 1800-2026
   - Neighborhood: Select from dropdown
3. **Click "Predict House Price"**
4. **View the predicted price**

## 🐛 Troubleshooting

**Model not loading?**
- Ensure `.pkl` files are in the `model/` folder
- Check file paths in `app.py`

**Import errors?**
- Run: `pip install -r requirements.txt`
- Use Python 3.8 or higher

**Deployment issues?**
- Ensure `gunicorn` is in requirements.txt
- Check that all `.pkl` files are uploaded
- Verify the start command: `gunicorn app:app`

## 📧 Support

If you encounter any issues:
1. Check the notebook outputs for errors
2. Verify all files are in correct locations
3. Ensure Python version compatibility
4. Review deployment platform logs

## 📄 License

This project is for educational purposes as part of a course assignment.

---

**Created by**: [Your Name]  
**Matric Number**: [Your Matric Number]  
**Date**: January 2026  
**Course**: Machine Learning / Data Science
# HousePrice_Project_Emilola_Divine_23CG034066

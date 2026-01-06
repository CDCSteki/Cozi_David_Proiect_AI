# Ames Housing Price Prediction

A complete machine learning pipeline for predicting house prices in Ames, Iowa, featuring exploratory data analysis, model comparison, hyperparameter tuning, and an interactive Streamlit web application.

## Key Features

- **Advanced EDA:** Comprehensive analysis of skewness, outliers, and feature correlations.
- **Robust Preprocessing:** Log-transformations for skewed data, missing value imputation, and One-Hot Encoding.
- **Feature Engineering:** Creation of domain-specific features like `TotalSF`, `TotalBathrooms`, and `TotalPorch`.
- **Hybrid Modeling:** A weighted ensemble of **Ridge, SVR, GradientBoosting, XGBoost, LightGBM, and RandomForest**.
- **Interactive UI:** A user-friendly web app built with Streamlit for real-time predictions.

## Installation & Setup

### Step 1: Get the Project Files

**Option A: With Git**
```bash
git clone https://github.com/CDCSteki/housePricingML.git
cd housePricingMl
```

**Option B: Without Git**
1. Click the green "Code" button on GitHub
2. Select "Download ZIP"
3. Extract the ZIP file to your desired location
4. Open terminal/command prompt in that folder

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Option A: Full Pipeline (EDA + Modeling)

### Step 1: Launch Jupyter Notebook (for running analysis)

```bash
jupyter notebook
```
This will open Jupyter in your browser where you can access the notebooks.

### Step 2: Run Exploratory Data Analysis

**Instructions:**

- Open the notebook in your browser (notebooks/EDA.ipynb)
- Run all cells: `Run → Run All Cells` or press `Shift + Enter` for each cell

**Output:**

- Cleaned dataset in `data/cleaned/`
- Visualizations in `plots/eda/`

### Step 3: Train Models

**Instructions:**

- Open the notebook in your browser (notebooks/model.ipynb)
- Run all cells: `Run → Run All Cells`

**Output:**

- Trained models in `models/`
- Performance metrics in `data/results/`
- Comparison plots in `plots/model_comp/`

### Option B: Use Pre-trained Model (Quick Start)

If models are already trained, skip to the web app:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Using the Web Application

1. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

2. **Enter property details** in the organized tabs:
   - 📍 Location & General Info
   - 📐 Areas & Layout
   - ✨ Quality & Condition
   - 🛁 Utilities & Extras

3. **Click "Calculate Price"** to get:
   - Estimated market value
   - Price per square foot
   - Contextual insights (new construction, premium neighborhood)

## Technical Details

### Data Processing Pipeline

1. **Missing Value Imputation**
   - Categorical: Filled with 'None' for logical absence
   - Numerical: Median imputation by neighborhood
   
2. **Outlier Removal**
   - Extreme values removed from LotArea, GrLivArea, BsmtFinSF1

3. **Skewness Correction**
   - Log/Sqrt transformation for continuous variables
   - Threshold: |skewness| > 0.75

4. **Feature Engineering**
   - `TotalSF`: Sum of basement + 1st + 2nd floor areas
   - `Total_Bathrooms`: Weighted sum of all bathrooms
   - `TotalPorch`: Combined porch areas
   - Binary indicators: haspool, has2ndfloor, hasgarage, etc.

5. **Multicollinearity Handling**
   - Removed: GarageYrBlt, TotRmsAbvGrd, GarageCars

### Models Evaluated

- Ridge Regression 
- Gradient Boosting
- XGBoost
- LightGBM
- Random Forest
- Support Vector Regression (SVR)
- Decision Tree
- K-Nearest Neighbors

### Ensemble Strategy

Weighted blend of top performers:

- Ridge: 20%
- SVR: 20%
- XGBoost: 20%
- LightGBM: 20%
- Gradient Boosting: 15%
- Random Forest: 5%

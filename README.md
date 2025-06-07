# World Happiness Report 2024: Analysis and Prediction

This project analyzes and predicts global happiness levels using the **World Happiness Report 2024** dataset. Built as an interactive **Streamlit application**, the project covers:
- Data exploration
- Statistical analysis and visualization
- Machine learning models to predict life satisfaction (Life Ladder scores)

Developed as a final project for the *Programming for Data Science* course at the University of Verona.

---

## Dataset Description

The dataset includes multiple years of data across 100+ countries, with features such as:

- **Life Ladder**: national happiness score (target variable)
- **Log GDP per capita**
- **Social support**
- **Healthy life expectancy at birth**
- **Freedom to make life choices**
- **Generosity**
- **Perceptions of corruption**
- **Positive and Negative affect measures**

---

## App Features

### 1. Data Exploration
- Dataset overview and description
- Handling missing data and imputation strategy
- Filtering by country and visualizing life expectancy trends

### 2. Data Analysis
- Bar and box plots of life expectancy
- Histograms of happiness (Life Ladder) and GDP
- Correlation heatmaps
- Country-level comparisons:
  - Happiest & unhappiest nations
  - Most generous vs. least generous
  - Time trend of happiness for the least happy country

### 3. Machine Learning
Two models were trained to predict **Life Ladder** using selected features:

- **Linear Regression**
  - RMSE (test): ~0.56  
  - R² Score (test): ~0.74

- **Random Forest Regression**
  - RMSE (test): ~0.41  
  - R² Score (test): ~0.86  
  - Outperformed linear regression in accuracy and robustness

Visual comparisons of actual vs. predicted values are included.

---

## Technologies Used

- Python (Pandas, NumPy, Scikit-learn)
- Streamlit (for interactive web app)
- Seaborn, Matplotlib (for data visualization)
- Linear Regression, Random Forest (from `sklearn`)


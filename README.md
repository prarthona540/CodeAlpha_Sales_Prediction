<div align="center">

# 📈 Sales Prediction using Python

### Predicting product sales from TV, Radio and Newspaper advertising budgets with Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

*CodeAlpha Data Science Internship · Task 4*

</div>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Dataset Information](#-dataset-information)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Machine Learning Models](#-machine-learning-models)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Workflow](#-project-workflow)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Future Improvements](#-future-improvements)
- [Screenshots](#-screenshots)
- [Requirements](#-requirements)
- [License](#-license)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Project Overview

### ❓ What is Sales Prediction?
Sales prediction is a **supervised regression** problem: the target (`Sales`) is a continuous number, and a model learns from historical examples where the answer is already known. Here, the model estimates how many units will be sold based on how much was spent on **TV**, **Radio** and **Newspaper** advertising.

### 💼 Why do businesses need it?
- **Smarter budgeting** – spend more where the return is highest.
- **Risk reduction** – test "what-if" scenarios before spending real money.
- **Better planning** – production, staffing and inventory depend on expected sales.
- **Competitive advantage** – data-driven companies react faster and waste less.

### 🧩 Problem Statement
Given the amount of money spent on advertising through **TV**, **Radio** and **Newspaper**, predict the number of units sold (the `Sales` column, in thousands of units).

Marketing budgets are limited, so a business needs to know:
- Which advertising channel drives sales the most?
- How many sales can we expect for a planned budget?
- Where should the next marketing dollar go?

### 🎯 Objectives
1. Clean and understand the Advertising dataset.
2. Explore relationships between each advertising channel and sales.
3. Train **Linear Regression**, **Decision Tree** and **Random Forest** models.
4. Evaluate them with **MAE, MSE, RMSE and R²**.
5. Select the best model and explain **which channels matter most**.
6. Predict sales for a new advertising plan and **save the model** for reuse.

### 🌍 Real-world Applications

| Domain | Use Case |
|--------|----------|
| 📣 Marketing | Budget allocation across channels |
| 🛒 Retail & E-commerce | Campaign planning and demand forecasting |
| 💰 Finance | Revenue forecasting and ROI estimation |
| 🏭 Supply Chain | Inventory planning from expected demand |
| 🚀 Startups | Deciding where limited ad money has the most impact |

---

## ✨ Features

✔ Automatic dataset detection, with an upload prompt when running in Google Colab
✔ Data inspection (`head`, `tail`, `sample`, `info`, `describe`)
✔ Data cleaning – missing values, duplicate rows, data types and negative-value checks
✔ Exploratory Data Analysis with histograms, KDE plots, box plots, pair plot, correlation heatmap, scatter plots and regression plots
✔ Skewness analysis of every variable
✔ Outlier detection using the IQR rule
✔ Feature selection (`TV`, `Radio`, `Newspaper` → `Sales`)
✔ 80/20 train-test split with `random_state=42`
✔ Multiple machine learning models – Linear Regression, Decision Tree, Random Forest
✔ Model evaluation with MAE, MSE, RMSE and R² Score
✔ Overfitting check (train vs test R²)
✔ Model comparison table and charts, with automatic best-model selection
✔ Feature importance analysis (Random Forest)
✔ Actual vs Predicted plot and residual analysis
✔ Sales prediction for a new advertising plan
✔ Model saving and reload verification with `joblib`
✔ Automatic generation of `requirements.txt`

---

## 🛠️ Technologies Used

| Category | Details |
|----------|---------|
| **Language** | Python |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `train_test_split`, `mean_absolute_error`, `mean_squared_error`, `r2_score`) |
| **Model Persistence** | Joblib |
| **Tools** | Jupyter Notebook |
| **Platform** | Google Colab ready · runs locally in Jupyter |

---

## 📂 Dataset Information

| Property | Details |
|----------|---------|
| **Dataset Name** | Advertising Dataset (`advertising.csv`) |
| **Dataset Source** | The classic Advertising dataset (a source link is not specified in the notebook) |
| **Number of Rows** | 200 (one row per market) |
| **Number of Columns** | 4 |
| **Input Features** | `TV`, `Radio`, `Newspaper` |
| **Target Variable** | `Sales` |

| Column | Type | Description |
|--------|------|-------------|
| `TV` | Feature | Advertising budget spent on TV |
| `Radio` | Feature | Advertising budget spent on Radio |
| `Newspaper` | Feature | Advertising budget spent on Newspaper |
| `Sales` | **Target** | Units sold in that market |

> Budgets are in thousands of dollars and sales are in thousands of units.

**Data quality:** no missing values, no duplicate rows, all columns are numeric (`float64`) and there are no negative values.

**Descriptive statistics**

| Column | Mean | Std | Min | 25% | 50% | 75% | Max |
|--------|-----:|----:|----:|----:|----:|----:|----:|
| TV | 147.04 | 85.85 | 0.7 | 74.38 | 149.75 | 218.82 | 296.4 |
| Radio | 23.26 | 14.85 | 0.0 | 9.98 | 22.90 | 36.52 | 49.6 |
| Newspaper | 30.55 | 21.78 | 0.3 | 12.75 | 25.75 | 45.10 | 114.0 |
| Sales | 15.13 | 5.28 | 1.6 | 11.00 | 16.00 | 19.05 | 27.0 |

---

## 📊 Exploratory Data Analysis

| # | Visualization | Insight |
|---|---------------|---------|
| 1 | **Histograms** (all 4 columns) | `TV` and `Radio` are spread fairly evenly across their ranges. `Newspaper` is right-skewed, and `Sales` is roughly bell-shaped around 15. |
| 2 | **Distribution plots (KDE) with mean line** | Skewness: TV −0.07, Radio 0.09, Sales −0.07 (nearly symmetric); Newspaper 0.89 (clear right tail). No transformation was required. |
| 3 | **Box plots** | `TV`, `Radio` and `Sales` show no outliers; `Newspaper` has a couple of points above the upper whisker. |
| 4 | **Pair plot** (with regression lines) | `TV` vs `Sales` shows a tight upward trend, `Radio` vs `Sales` is positive but more scattered, and `Newspaper` vs `Sales` is a near-flat cloud. |
| 5 | **Correlation heatmap** | Correlation with `Sales`: **TV 0.901**, **Radio 0.350**, **Newspaper 0.158**. TV is by far the strongest indicator. |
| 6 | **Each channel vs Sales** (scatter plots) | TV shows an almost linear increase; Radio trends upward with more variation; Newspaper shows no strong pattern. |
| 7 | **Sales distribution** (mean and median lines) | Mean ≈ 15 and median ≈ 16, so the target is approximately symmetric, and most markets sell between 11 and 19 (the IQR). |
| 8 | **Regression plots** (95% confidence band) | TV has a steep line and narrow band (reliable); Radio has a wider band; Newspaper is almost flat with a very wide band. |
| 9 | **Outlier detection** (IQR rule, box plot with highlighted points) | Only `Newspaper` has outliers – **2 markets** (rows 16 and 101). They are valid observations, so they were kept. |

---

## 🤖 Machine Learning Models

| Model | How it works | Configuration | Strength | Weakness |
|-------|--------------|---------------|----------|----------|
| 📏 **Linear Regression** | Fits a straight line/plane through the data | Default | Simple, fast, interpretable | Cannot capture curved or interaction effects |
| 🌳 **Decision Tree Regressor** | Splits data into branches using if/else rules | `random_state=42` (depth 13, 150 leaves) | Captures non-linear patterns | Can overfit and be unstable |
| 🌲 **Random Forest Regressor** | Averages many decision trees | `n_estimators=200`, `random_state=42` | Accurate and robust, reduces overfitting | Less interpretable, slower |

**Model comparison (test set, ranked by R²)**

| Rank | Model | MAE ⬇️ | MSE ⬇️ | RMSE ⬇️ | R² Score ⬆️ |
|:----:|-------|:------:|:------:|:-------:|:-----------:|
| 🥇 | **Random Forest** | **0.9140** | **1.3990** | **1.1828** | **0.9547** |
| 🥈 | Linear Regression | 1.2748 | 2.9078 | 1.7052 | 0.9059 |
| 🥉 | Decision Tree | 1.2400 | 3.4020 | 1.8445 | 0.8899 |

🏆 **Best-performing model: Random Forest Regressor**, selected automatically in the notebook as the model with the highest test R².

**Overfitting check (train vs test R²)**

| Model | Train R² | Test R² | Gap |
|-------|:--------:|:-------:|:---:|
| Linear Regression | 0.9001 | 0.9059 | −0.0058 |
| Decision Tree | 1.0000 | 0.8899 | 0.1101 |
| Random Forest | 0.9918 | 0.9547 | 0.0370 |

The Decision Tree overfits the training data, while Linear Regression generalises consistently and Random Forest keeps a high, stable test score.

**Linear Regression coefficients:** intercept `4.7141`, TV `0.0545`, Radio `0.1009`, Newspaper `0.0043`.

---

## 📏 Evaluation Metrics

All metrics are computed on the **20% held-out test set** (40 rows).

| Metric | Meaning | Better when |
|--------|---------|:-----------:|
| **MAE** – Mean Absolute Error | Average of \|actual − predicted\|, in the same units as `Sales` | ⬇️ Lower |
| **MSE** – Mean Squared Error | Average of (actual − predicted)²; penalises large errors more heavily | ⬇️ Lower |
| **RMSE** – Root Mean Squared Error | √MSE; error in the original units, sensitive to big mistakes | ⬇️ Lower |
| **R² Score** | Share of variance in `Sales` explained by the model (1 = perfect, 0 = no better than predicting the mean) | ⬆️ Higher |

**Actual values from the notebook**

| Model | MAE | MSE | RMSE | R² |
|-------|:---:|:---:|:----:|:--:|
| Linear Regression | 1.2748 | 2.9078 | 1.7052 | 0.9059 |
| Decision Tree | 1.2400 | 3.4020 | 1.8445 | 0.8899 |
| Random Forest | 0.9140 | 1.3990 | 1.1828 | 0.9547 |

---

## 🔄 Project Workflow

1. **Load the dataset** – locate `advertising.csv`, read it with Pandas and drop any leftover index column.
2. **Inspect the data** – `head`, `tail`, `sample`, shape, `info` and `describe`.
3. **Clean the data** – check missing values, remove duplicates and verify data types and negative values.
4. **Explore the data (EDA)** – histograms, KDE plots, box plots, pair plot, correlation heatmap, scatter and regression plots, sales distribution and IQR outlier detection.
5. **Select features** – `TV`, `Radio`, `Newspaper` as inputs and `Sales` as the target.
6. **Split the data** – 80% training (160 rows) and 20% testing (40 rows) with `random_state=42`.
7. **Train the models** – Linear Regression, Decision Tree Regressor and Random Forest Regressor.
8. **Evaluate the models** – MAE, MSE, RMSE and R² on the test set, plus a train vs test R² overfitting check.
9. **Compare the models** – results table, bar charts and automatic best-model selection.
10. **Analyse feature importance** – Random Forest feature importances.
11. **Analyse predictions** – Actual vs Predicted plot and residual analysis.
12. **Predict new sales** – forecast sales for a new advertising plan.
13. **Save the model** – export with `joblib` and verify by reloading it.

---

## 🗂️ Folder Structure

The notebook looks for `advertising.csv` in the working directory, so keep the dataset in the same folder as the notebook.

```text
sales-prediction-using-python/
│
├── 📄 Sales_Prediction_CodeAlpha.ipynb   # Full analysis notebook
├── 📄 advertising.csv                    # Dataset
├── 📄 sales_prediction_model.pkl         # Saved best model (generated by the notebook)
├── 📄 requirements.txt                   # Python dependencies
├── 📄 README.md                          # Project documentation
└── 📄 LICENSE                            # MIT License
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/sales-prediction-using-python.git
cd sales-prediction-using-python
```

**2. (Optional) Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install the dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option A – Locally with Jupyter
1. Make sure `advertising.csv` is in the **same folder** as the notebook.
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `Sales_Prediction_CodeAlpha.ipynb`.
4. Click **Kernel ▸ Restart & Run All**.
5. When it finishes, `sales_prediction_model.pkl` and `requirements.txt` are created in the same folder.

### Option B – Google Colab
1. Upload `Sales_Prediction_CodeAlpha.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Click **Runtime ▸ Run all**.
3. When prompted, upload `advertising.csv`.
4. The saved model and `requirements.txt` are downloaded automatically at the end.

### 🔮 Using the saved model
```python
import joblib
import pandas as pd

model = joblib.load("sales_prediction_model.pkl")
new_plan = pd.DataFrame({"TV": [230], "Radio": [37], "Newspaper": [69]})
print(model.predict(new_plan))   # ≈ 21.81
```

---

## 🏆 Results

| Key Finding | Detail |
|-------------|--------|
| 🧹 **Clean dataset** | 200 rows, no missing values, no duplicates |
| 🏆 **Best model** | Random Forest – R² = 0.9547 (explains about 95.5% of the variation in Sales) |
| 🎯 **Prediction error** | MAE = 0.9140 (about 914 units on average), RMSE = 1.1828 |
| 📺 **TV is the strongest driver** | Correlation with Sales 0.901; feature importance 84.4% |
| 📻 **Radio is a secondary driver** | Correlation 0.350; feature importance 13.6% |
| 📰 **Newspaper adds very little** | Correlation 0.158; feature importance 1.9% |

**Feature importance (Random Forest)**

| Feature | Importance |
|---------|:----------:|
| TV | 0.8444 |
| Radio | 0.1364 |
| Newspaper | 0.0192 |

**Residual analysis (Random Forest, test set):** mean residual −0.2206, standard deviation 1.1768, largest over-prediction −3.70 and largest under-prediction 3.53. Residuals scatter around zero without an obvious pattern.

**Example prediction** – plan: TV = 230, Radio = 37, Newspaper = 69

| Model | Predicted Sales |
|-------|:---------------:|
| **Random Forest (best)** | **21.81** (≈ 21,811 units) |
| Linear Regression | 21.29 |
| Decision Tree | 22.10 |

The saved model was reloaded and produced an identical prediction (21.81).

**💡 Business insight:** budget is best focused on **TV + Radio**, and newspaper spending is worth reviewing, since the model gives it very little weight.

**⚠️ Limitations (as noted in the notebook)**
- Small dataset (200 rows), so results may change with more data.
- Only three input features; real sales are also influenced by price, season, competition and more.
- Models were not hyperparameter-tuned, and results come from a single train/test split.
- The model shows association, not proven causation.

---

## 🚀 Future Improvements

| Idea | Description |
|------|-------------|
| 🎛️ **Hyperparameter Tuning** | `GridSearchCV` / `RandomizedSearchCV` / Optuna to tune tree depth, number of trees, etc. |
| 🔁 **Cross Validation** | K-fold CV for a more reliable estimate than a single split |
| ⚡ **XGBoost** | Gradient boosting often beats Random Forest on tabular data |
| 💡 **LightGBM** | Fast, memory-efficient boosting |
| 🧪 **Feature Engineering** | Interaction terms (e.g. `TV × Radio`), total ad spend, log or polynomial features |
| 🖥️ **Streamlit App** | Interactive dashboard where managers enter budgets and see predictions |
| 🌐 **Flask / FastAPI API** | Serve the model as a REST endpoint |
| ☁️ **Cloud Deployment** | Deploy on Render, Hugging Face Spaces, AWS, GCP or Azure |
| 📈 **Monitoring** | Track model drift and retrain when new sales data arrives |

---

## 🖼️ Screenshots

### 📋 Dataset Preview
![Dataset Preview](images/dataset_preview.png)

### 🔥 Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)

### 📺 Scatter Plot
![Scatter Plot](images/scatter_plot.png)

### 🌟 Feature Importance
![Feature Importance](images/feature_importance.png)

### 🔍 Actual vs Predicted
![Actual vs Predicted](images/actual_vs_predicted.png)

> 📌 Export these charts from the notebook, save them in an `images/` folder, and they will appear here.

---

## 📦 Requirements

Save the following as `requirements.txt` (the notebook also generates this file automatically):

```text
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
joblib>=1.2.0
jupyter>=1.0.0
```

---

## 📜 License

This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2026 Prarthona Biswas Puja

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👩‍💻 Author

**Prarthona Biswas Puja**
🎓 Software Engineering Student
🏢 CodeAlpha Data Science Intern

| | |
|---|---|
| 🐙 **GitHub** | [github.com/your-username](https://github.com/your-username) |
| 💼 **LinkedIn** | [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile) |

---

## 🙏 Acknowledgements

- **CodeAlpha** – for the Data Science Internship and the project task
- **Scikit-learn** – machine learning models and evaluation metrics
- **Pandas** – data loading and manipulation
- **NumPy** – numerical computing
- **Matplotlib** – data visualization
- **Seaborn** – statistical visualization

---

<div align="center">

⭐ *If you found this project useful, please consider giving it a star!* ⭐

</div>

# Ayush Choudhary — Celebal Technologies Data Science Internship

This repository contains all weekly assignment notebooks submitted as part of the **Celebal Technologies Data Science Internship** program (8 Weeks).

---

## Intern Details

| Field | Details |
|-------|---------|
| **Name** | Ayush Choudhary |
| **Program** | Data Science Internship |
| **Organization** | Celebal Technologies |
| **Duration** | 8 Weeks |
| **Email** | ayushchoudhary18481@gmail.com |

---

## Weekly Assignment Progress

| Week | Notebook | Topic | Status |
|------|----------|-------|--------|
| Week 1 | `week1_Ayush Choudhary.ipynb` | Python & Data Analysis Fundamentals | ✅ Submitted |
| Week 2 | `week2_Ayush Choudhary.ipynb` | End-to-End ML Pipeline (Sales/Price Data) | ✅ Submitted |
| Week 3 | `week3_Ayush Choudhary.ipynb` | — | 🔜 Upcoming |
| Week 4 | `week4_Ayush Choudhary.ipynb` | — | 🔜 Upcoming |
| Week 5 | `week5_Ayush Choudhary.ipynb` | — | 🔜 Upcoming |
| Week 6 | `week6_Ayush Choudhary.ipynb` | — | 🔜 Upcoming |
| Week 7 | `week7_Ayush Choudhary.ipynb` | — | 🔜 Upcoming |
| Week 8 | `week8_Ayush Choudhary.ipynb` | — | 🔜 Upcoming |

---

## Repository Structure

```
Ayush Choudhary Celebal Assignments/
│
├── week1_Ayush Choudhary.ipynb             # Week 1 — Python & Data Analysis Fundamentals
├── week2_Ayush Choudhary.ipynb             # Week 2 — End-to-End ML Pipeline on Sales/Price Data
├── week3_Ayush Choudhary.ipynb             # Week 3 — (upcoming)
├── week4_Ayush Choudhary.ipynb             # Week 4 — (upcoming)
├── week5_Ayush Choudhary.ipynb             # Week 5 — (upcoming)
├── week6_Ayush Choudhary.ipynb             # Week 6 — (upcoming)
├── week7_Ayush Choudhary.ipynb             # Week 7 — (upcoming)
├── week8_Ayush Choudhary.ipynb             # Week 8 — (upcoming)
├── tesla_deliveries_dataset_2015_2025.csv  # Dataset for Week 2 ML Pipeline
└── README.md                               # This file
```

---

## Week 2 — End-to-End Machine Learning Pipeline on Sales/Price Data

> **Note:** The notebook `week2_Ayush Choudhary.ipynb` implements a complete machine learning pipeline on sales and price data.

### Dataset

**Tesla Deliveries and Production Dataset (2015–2025)**

- **Records:** 2,640 rows
- **Features:** 12 columns

| Column | Description |
|--------|-------------|
| Year | Year of record |
| Month | Month of record |
| Region | Geographic region (North America, Europe, Asia, Middle East) |
| Model | Tesla vehicle model (Model S, 3, X, Y, Cybertruck) |
| Estimated_Deliveries | Number of vehicles delivered |
| Production_Units | Number of vehicles produced |
| Avg_Price_USD | Average selling price in USD |
| Battery_Capacity_kWh | Battery capacity in kilowatt-hours |
| Range_km | Vehicle range in kilometers |
| CO2_Saved_tons | Estimated CO2 savings in tons |
| Source_Type | Data source reliability (Official / Interpolated / Estimated) |
| Charging_Stations | Number of nearby charging stations |

### Pipeline Stages

#### 1. Data Loading and Inspection
- Shape, columns, data types, descriptive statistics

#### 2. Data Cleaning
- Missing values — none found
- Duplicate rows — none found
- Encoded `Source_Type` as ordinal reliability score

#### 3. Exploratory Data Analysis (EDA)
- Deliveries by Model, Region, Year
- Price distribution
- Correlation heatmap
- Monthly seasonality
- CO2 savings trend

#### 4. Feature Engineering
| Feature | Description |
|---------|-------------|
| Quarter | Fiscal quarter (1–4) |
| Delivery_Rate | Deliveries / Production Units |
| CO2_per_Delivery | CO2 saved per vehicle |
| Price_per_km | Price per km of range |
| km_per_kWh | Range efficiency |
| Log_Charging | Log of charging station count |
| Model_Age | Years since 2015 |

#### 5. Regression Modeling
Six models compared — Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting

#### 6. Hyperparameter Tuning
GridSearchCV with 5-fold cross-validation on Random Forest

#### 7. Time Series Forecasting
SARIMA(1,1,1)(1,1,1)[12] — ADF test, seasonal decomposition, ACF/PACF, 12-month future forecast

---

## Setup & Installation

### Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels
```

Install everything:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

### Run on Windows (VS Code)

1. Open the folder in VS Code
2. Select a Python 3 interpreter (Ctrl+Shift+P → "Select Interpreter")
3. Open any `.ipynb` file
4. Click **Run All** at the top of the notebook

### Run with JupyterLab

```bash
pip install jupyterlab
jupyter lab
```

---

## Coding Style

All notebooks follow a consistent style:

- **Markdown cell above every code cell** — explains what the code does
- **No inline comments inside code** — kept clean and readable
- Code cells perform one focused task each
- Section headers: `## Section Name`
- EDA labels: `**EDA 1 — Description**`

---

## Submission Checklist

- [x] Week 1 notebook complete and submitted
- [x] Week 2 notebook complete and submitted
- [ ] Week 3 notebook — upcoming
- [ ] Week 4 notebook — upcoming
- [ ] Week 5 notebook — upcoming
- [ ] Week 6 notebook — upcoming
- [ ] Week 7 notebook — upcoming
- [ ] Week 8 notebook — upcoming

---

## Contact

**Ayush Choudhary**
Email: ayushchoudhary18481@gmail.com
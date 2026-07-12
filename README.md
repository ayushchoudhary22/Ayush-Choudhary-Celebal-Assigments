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

| Week | Notebook / Folder | Topic | Status |
|------|-------------------|-------|--------|
| Week 1 | `week1_Ayush Choudhary.ipynb` | Python & Data Analysis Fundamentals | ✅ Submitted |
| Week 2 | `week2_Ayush Choudhary.ipynb` | End-to-End ML Pipeline (Sales/Price Data) | ✅ Submitted |
| Week 3 | `week3_Ayush Choudhaer.ipynb` | Unsupervised Learning — Country Clustering (K-Means, DBSCAN, PCA) | ✅ Submitted |
| Week 4 | `week4_Ayush_Choudhary.ipynb` | CIFAR-10 Image Classification (ANN vs CNN) | ✅ Submitted |
| Week 5 | `week5_Ayush Choudhary.ipynb` | Text Generation using Vanilla RNN, LSTM, and GRU | ✅ Submitted |
| Week 6 | `week6_Ayush Choudhary.ipynb` | Autoencoder for Image Denoising — MNIST (PyTorch) | ✅ Submitted |
| Week 7 | [`week7/`](file:///c:/Users/Ayush%20choudhary/Desktop/Ayush%20Choudhary%20Celebal%20Assigments/week7) | Document Question Answering System (RAG) (Streamlit) | ✅ Submitted |
| Week 8 | `week8_Ayush Choudhary.ipynb` | — | 🔜 Upcoming |

---

## Repository Structure

```
Ayush Choudhary Celebal Assignments/
│
├── week1_Ayush Choudhary.ipynb             # Week 1 — Python & Data Analysis Fundamentals
├── week2_Ayush Choudhary.ipynb             # Week 2 — End-to-End ML Pipeline on Sales/Price Data
├── week3_Ayush Choudhaer.ipynb             # Week 3 — Country Clustering: K-Means, DBSCAN, PCA
├── week4_Ayush_Choudhary.ipynb             # Week 4 — CIFAR-10 Image Classification (ANN vs CNN)
├── week5_Ayush Choudhary.ipynb             # Week 5 — Text Generation: Simple RNN, LSTM, GRU
├── week6_Ayush Choudhary.ipynb             # Week 6 — Autoencoder for Image Denoising (PyTorch)
├── week7/                                  # Week 7 — Document Question Answering System (RAG)
│   ├── data/                               #   ├── Local PDF storage
│   ├── app.py                              #   ├── Streamlit application UI
│   ├── chatbot.py                          #   ├── Extractive QA logic & answer synthesis helper
│   ├── vectorstore.py                      #   ├── TF-IDF Vectorizer & BM25 hybrid search
│   ├── run.bat                             #   └── Single-click launcher batch script
├── tesla_deliveries_dataset_2015_2025.csv  # Dataset for Week 2 ML Pipeline
├── Country-data.csv                        # Dataset for Week 3 Country Clustering
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

## Week 3 — Unsupervised Learning: Country Clustering

> **Notebook:** `week3_Ayush Choudhaer.ipynb`

### Dataset

**Country-data.csv** — Socio-economic and health indicators for 167 countries

- **Records:** 167 rows
- **Features:** 10 columns

| Column | Description |
|--------|-------------|
| country | Country name |
| child_mort | Child mortality rate per 1000 live births |
| exports | Exports as % of GDP |
| health | Health spending as % of GDP |
| imports | Imports as % of GDP |
| income | Net income per person (USD) |
| inflation | Annual inflation rate (%) |
| life_expec | Life expectancy (years) |
| total_fer | Total fertility rate |
| gdpp | GDP per capita (USD) |

### Pipeline Stages

#### 1. Data Loading & Exploration
- Loaded 167 × 10 dataset, verified shape and data types

#### 2. Data Cleaning
- Stripped whitespace from column names
- Dropped duplicate records
- Forced numeric types on all non-country columns
- Imputed missing values using column-wise median

#### 3. Feature Scaling
- Dropped `country` identifier column
- Applied `StandardScaler` to all 9 numeric features

#### 4. K-Means Clustering (Elbow Method)
- Tested k ∈ [2, 10], recorded inertia values
- Plotted Elbow curve + Silhouette score curve
- Selected **best_k = 3** based on elbow inflection

#### 5. Clustering Results

| Cluster | Label | Countries | Child Mortality | Life Expectancy | GDP per Capita |
|---------|-------|-----------|----------------|-----------------|----------------|
| Cluster 0 | Developed | 36 | 5.00 | 80.13 yrs | $42,494 |
| Cluster 1 | Underdeveloped | 47 | 92.96 | 59.19 yrs | $1,922 |
| Cluster 2 | Developing | 84 | 21.93 | 72.81 yrs | $6,486 |

- **Silhouette Score: 0.2833**

#### 6. DBSCAN Clustering
- Parameters: `eps=1.5`, `min_samples=5`
- Identified **30 noise/outlier** countries

#### 7. PCA Visualization
- Reduced 9D scaled features → 2D via PCA
- Plotted color-coded K-Means and DBSCAN scatterplots side by side

#### 8. Classification (Extended)
- Used K-Means labels as classification targets
- **Random Forest** Accuracy: **100%**
- **XGBoost** Accuracy: **97.06%**

#### 9. Key Socio-Economic Observations (Section 14)
1. **High-Mortality Cluster** — Cluster 1 shows child_mort=92.96 and gdpp=$1,922, indicating severe underdevelopment (Sub-Saharan Africa/South Asia)
2. **Top-Tier Economic Zones** — Cluster 0 shows near-zero child mortality (5.00) and life expectancy >80 years (Western Europe/North America/Australia)
3. **Developing Nations** — Cluster 2 (84 countries) reflects transitional economies in Latin America, North Africa, and Southeast Asia
4. **DBSCAN Outliers** — 30 nations with extreme characteristics (oil-rich Gulf states, conflict zones, island microstates)
5. **Key Predictors** — child_mort, gdpp, life_expec, and income are the strongest development indicators

---

## Week 4 — CIFAR-10 Image Classification (ANN vs CNN)

> **Notebook:** `week4_Ayush_Choudhary.ipynb`

### Dataset

**CIFAR-10** — Standard image classification database
- **Images:** 60,000 color images (50,000 Train, 10,000 Test)
- **Dimensions:** 32×32×3 pixels (RGB)
- **Classes:** 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

### Model Architectures & Results

We built and evaluated four different deep learning configurations:

| Model | Architecture Highlights | Epochs | Test Accuracy |
|-------|------------------------|--------|---------------|
| **Baseline ANN** | Flat vectors, Dense(512), Dropout(0.3), Dense(256) | 10 | **41.74%** |
| **Baseline CNN** | Conv2D(32, 64, 128), BatchNorm, MaxPooling2D | 10 | **64.16%** |
| **Upgraded ANN** | Dense(1024), Dropout(0.3), Dense(512), Dense(256) | 20 (EarlyStopping) | **39.73%** |
| **Augmented CNN** | Data Augmentation (Flip, Rotation, Zoom), Conv2D(32, 64, 128), BatchNorm | 20 (EarlyStopping) | **68.99%** |

### Key Observations
1. **Spatial Representation:** The CNN architectures significantly outperform the ANN models, showing that preserving 2D spatial features is crucial for image classification.
2. **ANN Overfitting:** Scaling up the ANN to wider layers (Upgraded ANN) led to overfitting, resulting in slightly lower test performance (39.73% vs 41.74%).
3. **Data Augmentation Benefit:** Adding random image transformations and training with EarlyStopping enabled the Augmented CNN to reach the highest accuracy (**68.99%**).

---

## Week 5 — Text Generation using Simple RNN, LSTM, and GRU

> **Notebook:** `week5_Ayush Choudhary.ipynb`

### Task Description
Develop and train sequence modeling architectures to learn grammatical structure and context from a custom deep learning text corpus, and evaluate text generation performance.

### Configuration & Hyperparameters
- **Custom Corpus:** 17-line detailed deep learning text database.
- **Embedding Dimensions:** 64
- **Hidden Units:** 128
- **Epochs:** 200 (optimized sequence transduction)

### Key Observations
1. **Vanishing Gradients:** Vanilla RNN learned short-range patterns but struggled to maintain contextual memory for complex sentences.
2. **Gated Architectures:** Both LSTM and GRU outperformed the basic RNN, showing stable training loss reduction and generating grammatical, meaningful text sequences.
3. **Efficiency:** GRU trained faster than LSTM due to fewer gate parameters while achieving comparable prediction quality.

---

## Week 6 — Autoencoder for Image Denoising (MNIST)

> **Notebook:** `week6_Ayush Choudhary.ipynb`

### Task Description
Build and train a deep learning architecture using **PyTorch** to perform image denoising on the MNIST handwritten digit database.

### Architecture Highlights
* **Encoder**: Convolutional layers (`Conv2d`, `ReLU`, `MaxPool2d`) that compress the input image of size $28 \times 28 \times 1$ into a low-dimensional latent space.
* **Decoder**: Transposed convolutional layers (`ConvTranspose2d`, `ReLU`, `Sigmoid`) that reconstruct the original image from the compressed latent vector, filtering out added Gaussian noise.
* **Loss Function**: Mean Squared Error (MSE) loss, directly comparing the reconstructed clean pixels with the target clean pixels.

---

## Week 7 — Document Question Answering System (RAG)

> **Folder:** `week7/`

### Task Description
Implement a complete **Retrieval-Augmented Generation (RAG)** chatbot interface that operates **100% offline**, allowing users to upload and index custom PDFs (like resumes, guides, notes) and query them with automatic citation highlights.

### Architectural Features
* **TF-IDF Hashing Vectorizer (NumPy)**: Custom implementation using stop-word filtering, unigrams/bigrams, and sublinear TF scaling.
* **BM25 Lexical Scorer**: Matches exact keywords to score search terms that only appear in specific target documents.
* **Targeted Section Extraction & Flow Parser**: Synonyms detect intent (projects, experience, education, skills, contact) and isolate sections using a forward-decaying flow algorithm that resets when hitting a different section header.
* **Context Boosting**: Document-level max scoring pulls in nearby chunks of matched target documents.

---

## Setup & Installation

### Core Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels
xgboost
torch
torchvision
pypdf
streamlit
google-generativeai
```

Install everything:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels xgboost torch torchvision pypdf streamlit google-generativeai
```

### Run Jupyter Notebooks (Weeks 1-6)

1. Open the folder in VS Code
2. Select a Python 3 interpreter (Ctrl+Shift+P → "Select Interpreter")
3. Open any `.ipynb` file and click **Run All**

### Run Streamlit App (Week 7)

Simply double-click or run the batch launcher inside the `week7` directory:
```bash
cd week7
.\run.bat
```

---

## Coding Style

All notebooks and project folders follow a consistent style:

- **Markdown cell / comment blocks above code** — explains the step-by-step logic
- **No inline comments inside code** — kept clean and readable
- Code cells perform one focused task each
- Section headers: `## Section Name`
- EDA labels: `**EDA 1 — Description**`

---

## Submission Checklist

- [x] Week 1 notebook complete and submitted
- [x] Week 2 notebook complete and submitted — End-to-End ML Pipeline (Sales/Price Data)
- [x] Week 3 notebook complete and submitted — Country Clustering (K-Means, DBSCAN, PCA) | Silhouette: 0.2833
- [x] Week 4 notebook complete and submitted — CIFAR-10 Classification (ANN vs CNN) | Augmented CNN: 68.99%
- [x] Week 5 notebook complete and submitted — Text Generation: RNN vs LSTM vs GRU
- [x] Week 6 notebook complete and submitted — Autoencoder for MNIST Denoising (PyTorch)
- [x] Week 7 project complete and submitted — Document QA System (RAG) (Streamlit)
- [ ] Week 8 notebook — upcoming

---

## Contact

**Ayush Choudhary**
Email: ayushchoudhary18481@gmail.com
GitHub: [ayushchoudhary22](https://github.com/ayushchoudhary22)
LinkedIn: [Ayush Choudhary](https://linkedin.com)
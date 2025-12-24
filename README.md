# 💼 LGD Prediction Models
## 🏷️ Tags  
- Models: Logistic Regression, XGBoost, Neural Network  
- Areas: Credit Risk, Econometrics, Loss Given Default (LGD), Machine Learning  

## 💡 About  
This project explores the prediction of Loss Given Default — a key component of credit risk measurement alongside Probability of Default and Exposure at Default.
The analysis combines traditional econometric models with modern machine learning algorithms estimates potencial losses after borrower default.
The study uses real-world loan-level data containing information on repayments, balances, and borrower characteristics.
The project also discusses the challenges of applying black-box models in regulated banking environment, where explainability remains a key factor

## 📂 Content  
- [**Main.py**](https://github.com/dzima22/LGD_Prediction_Models/tree/main/code/Main.py) — central pipeline setup covering data loading, preprocessing, modeling, evaluation and visualization ⚙️  
- [**modules**](https://github.com/dzima22/LGD_Prediction_Models/tree/main/code/modules) — folder containing separate modules for data processing, feature selection, modeling, evaluation, and visualizations 🧩  
- [**datasets**](https://github.com/dzima22/LGD_Prediction_Models/tree/main/datasets) — datasets used in analysis 📁  
- [**Research**](https://github.com/dzima22/LGD_Prediction_Models/blob/main/Reserch.pdf) —  full research report summarizing methodology, models, and findings 📄 
- [**visuals_folder**](https://github.com/dzima22/LGD_Prediction_Models/tree/main/visuals_folder) — output containing visuals and models' performance  📊  

## 🔬 Methodology  
- The analysis is based on **three datasets**: main loan data, monthly balance sheets, and repayment histories.  
- Data preprocessing includes cleaning, merging, feature transformation, and calculation of LGD-related variables.  
- Multiple models are trained and their results are compared:
  - **Logistic Regression** — baseline econometric model
  - **GLM (Quasi-binomial)** — logistic regression for bounded targets (e.g., RR/LGD in [0,1])
  - **XGBoost** — tree-based machine learning approach  
  - **Neural Network** — nonlinear architecture for LGD prediction  

## 📊 Findings  
Preliminary results indicate that Neural Network models outperform traditional econometric approaches in predicting LGD based on this dataset. However, there are limitations to using black-box models in decision-making cases, such as loan approval. 
Although, as black-box approaches have become more widespread, these limitations have been mitigated over time. 

## ⚙️ How to use
python code/Main.py

## Data
Due to size constraints, datasets are not stored in the repository.
See datasets/README.md for access instructions.

## Configuration
Global paths, constants, feature lists, and model hyperparameters are defined in `modules/config.py`.

## Project structure
```text
LGD_Prediction_Models/
├── README.md
├── requirements.txt
├── .gitignore
├── code/
│   ├── Main.py
│   ├── config.py
│   └── modules/
│       ├── __init__.py
│       ├── Data_processing.py
│       ├── feature_selection.py
│       ├── Models.py
│       ├── Evaluation.py
│       └── Visuals.py
├── datasets/
│   └── README.md
├── visuals_folder/
└── Research.pdf

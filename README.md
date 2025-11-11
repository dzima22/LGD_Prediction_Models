# 💼 LGD Prediction Models
## 🏷️ Tags  
- Models: Logistic Regression, XGBoost, Neural Network  
- Areas: Credit Risk, Econometrics, Loss Given Default (LGD), Machine Learning  

## 💡 About  
The project focuses on **Loss Given Default** prediction — one of the three key components of credit risk modeling (alongside PD and EAD).  
The goal is to estimate the proportion of exposure a lender loses in the event of borrower default, using both traditional statistical and modern machine learning approaches.  
The study utilizes loan-level data containing information on repayment behavior, balances, and loan characteristics.  

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

## ⚙️ How to Use  
1. Make sure you have downloaded the dataset folder(see the link, that is provided in the **Content** section).  
2. Run the main pipeline: python Main.py

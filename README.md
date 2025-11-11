# LGD_Prediction_Models
<div align="center">
  <img src="https://github.com/dzima22/The-evaluation-of-the-risk-of-corporate-bankruptcy-/blob/main/imgs/bankruptcy.jpg" alt="" width="250"/>
</div>

# ©️ Tags
- Models: logistic regression, a linear probability regression
- Areas: binomial models, Econometrics, regression, prediction, bankruptcy

# :bulb: About
The study focuses on the prediction of company bankruptcies using econometric techniques, specifically analyzing the financial data of Polish companies from 2008 to 2020. The goal is to evaluate the accuracy of logistic regression models in assessing the risk of bankruptcy.
# :open_file_folder: Content
- [research](https://github.com/dzima22/The-evaluation-of-the-risk-of-corporate-bankruptcy-/blob/main/research.pdf) -  main file containing the research ✍️
- [code_for_research](https://github.com/dzima22/The-evaluation-of-the-risk-of-corporate-bankruptcy-/blob/main/code_for_research.R) – analysis, modeling procedure stored as R script ⚗️
- [dataset](https://github.com/dzima22/The-evaluation-of-the-risk-of-corporate-bankruptcy-/blob/main/dane_do_projektu1.xlsx) – raw dataset 📀

# :test_tube: Methodology
- The study utilizes financial data of Polish companies listed on the Warsaw Stock Exchange (GPW) and NewConnect from 2008 to 2020. The data includes both bankrupt and non-bankrupt companies, categorized based on specific financial criteria. The dataset comprises financial statements and key financial ratios such as `Return on Assets (ROA)`, `Debt Ratio (OZ)`, `Current Ratio (BP)`, `Debt Service Coverage Ratio (WOZ)`, and `Total Asset Efficiency Ratio (WO)`. 🔍
- The sample is divided into a training set and a test set. The training set is used to build and calibrate the logistic regression models, while the test set is used to validate the model's predictive accuracy.

# 🏆 Findings

<div align="center">
  <img src="https://github.com/dzima22/The-evaluation-of-the-risk-of-corporate-bankruptcy-/blob/main/imgs/Rcurve.png" alt="" width="600"/>
</div>

- ROC Curve and AUC: The Receiver Operating Characteristic (ROC) curve is plotted to visualize the trade-off between sensitivity and specificity for different threshold values
- Area Under the Curve (AUC) is calculated to quantify the overall ability of the model to discriminate between the classes. So in this research `auc=0,94` so the predictive properties can be assessed as good, but it should be remembered that the study was conducted on a relatively small sample.

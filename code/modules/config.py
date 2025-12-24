
# Data files
loans_path = r"datasets/main_loan_base.csv"
balance_path = r"datasets/monthly_balance_base.csv"
repayment_path = r"datasets/repayment_base.csv"

# General settings
Random_state = 42
Test_size = 0.2
# Model feautures
features_model1 = ['collateral_value', 'avg_balance_amt', 'missed_repayments','interest', 'tenure_years', 'vintage_in_months','number_of_loans', 'cheque_bounces', 'avg_rr_per_city']
features_model2 = ['missed_repayments', 'cheque_bounces','number_of_loans', 'vintage_in_months']

numeric_features = [
    'loan_amount', 'collateral_value', 'cheque_bounces',
    'number_of_loans', 'missed_repayments', 'vintage_in_months',
    'tenure_years', 'interest', 'monthly_emi',
    'total_repayment', 'avg_balance_amt', 'avg_rr_per_city']
# Model parameters
pbounds_xgboost = {'n_estimators': (100, 300),'gamma': (1, 10),'learning_rate': (0.01, 0.1),'colsample_bytree': (0.5, 0.9),'reg_lambda': (1, 10),'reg_alpha': (1, 10),'subsample': (0.5, 0.8),'min_child_weight': (5, 20)}
eval_metric_xgboost='logloss'
cv_xgboost=3
epochs_nn_model=200
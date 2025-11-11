import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def logit_for_PC(df):
    features_model1 = ['collateral_value', 'avg_balance_amt', 'missed_repayments','interest', 'tenure_years', 'vintage_in_months','number_of_loans', 'cheque_bounces', 'avg_rr_per_city']
    features_model2 = ['missed_repayments', 'cheque_bounces','number_of_loans', 'vintage_in_months']
    X_train1, X_test1, y_train1, y_test1 = train_test_split(df[features_model1], df['PC'], test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(df[features_model2], df['PC'], test_size=0.2, random_state=42)
    logit_model1 = sm.Logit(y_train1, sm.add_constant(X_train1)).fit(disp=0)
    logit_model2 = sm.Logit(y_train2, sm.add_constant(X_train2)).fit(disp=0)
    return {"model1": {"model": logit_model1,"X_train": sm.add_constant(X_train1),"X_test": sm.add_constant(X_test1),"y_train": y_train1,"y_test": y_test1},
           "model2": {"model": logit_model2,"X_train": sm.add_constant(X_train2),"X_test": sm.add_constant(X_test2),"y_train": y_train2,"y_test": y_test2}}
def logit_model_adasyn(df):
    features_model1 = ['collateral_value', 'avg_balance_amt', 'missed_repayments','interest', 'tenure_years', 'vintage_in_months','number_of_loans', 'cheque_bounces', 'avg_rr_per_city']
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features_model1] = scaler.fit_transform(df[features_model1])
    X_train, X_test, y_train, y_test = train_test_split(df_scaled[features_model1],df_scaled['PC'],test_size=0.2,random_state=42)
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    X_resampled_const = sm.add_constant(X_resampled)
    X_test_const = sm.add_constant(X_test)
    logit_model = sm.Logit(y_resampled, X_resampled_const).fit(disp=0)
    return {"model": logit_model,"X_train": X_resampled_const,"y_train": y_resampled,"X_test": X_test_const,"y_test": y_test}
def glm_rr_models(df):
    rr_target = df['RR'].apply(lambda x: 0 if x <= 0 else (1 if x >= 1 else x))
    features_full = ['loan_amount','collateral_value','avg_balance_amt','interest','vintage_in_months','number_of_loans','missed_repayments','avg_rr_per_city','tenure_years','cheque_bounces']
    features_small = ['avg_balance_amt', 'vintage_in_months', 'number_of_loans','missed_repayments', 'cheque_bounces']
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    glm_full = sm.GLM(rr_target.loc[train_idx], sm.add_constant(df.loc[train_idx, features_full]), family=sm.families.Binomial())
    res_full = glm_full.fit()
    glm_small = sm.GLM(rr_target.loc[train_idx], sm.add_constant(df.loc[train_idx, features_small]), family=sm.families.Binomial())
    res_small = glm_small.fit()
    return {"model1": {"model":  res_full,"X_test": sm.add_constant(df.loc[test_idx,  features_full]),"y_test": rr_target.loc[test_idx]},
            "model2": {"model":  res_small,"X_test": sm.add_constant(df.loc[test_idx,  features_small]),"y_test": rr_target.loc[test_idx]}}
def xgb_models(df):
    features_model1 = ['collateral_value', 'avg_balance_amt', 'missed_repayments','interest', 'tenure_years', 'vintage_in_months','number_of_loans', 'cheque_bounces', 'avg_rr_per_city']
    features_model2 = ['missed_repayments', 'cheque_bounces','number_of_loans', 'vintage_in_months']
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    X_train1 = df.loc[train_idx, features_model1]
    X_test1  = df.loc[test_idx,  features_model1]
    y_train1 = df['PC'].loc[train_idx]
    y_test1  = df['PC'].loc[test_idx]
    scale_pos_weight1 = y_train1.value_counts()[0] / y_train1.value_counts()[1]
    X_train2 = df.loc[train_idx, features_model2]
    X_test2  = df.loc[test_idx,  features_model2]
    y_train2 = df['PC'].loc[train_idx]
    y_test2  = df['PC'].loc[test_idx]
    scale_pos_weight2 = y_train2.value_counts()[0] / y_train2.value_counts()[1]
    pbounds = {'n_estimators': (100, 300),'gamma': (1, 10),'learning_rate': (0.01, 0.1),'colsample_bytree': (0.5, 0.9),'reg_lambda': (1, 10),'reg_alpha': (1, 10),'subsample': (0.5, 0.8),'min_child_weight': (5, 20)}
    def xgb_cv_model1(n_estimators, gamma, learning_rate, colsample_bytree,reg_lambda, reg_alpha, subsample, min_child_weight):
        model = XGBClassifier(n_estimators=int(n_estimators),gamma=gamma,
            learning_rate=learning_rate,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            min_child_weight=min_child_weight,
            scale_pos_weight=scale_pos_weight1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42)
        scores = cross_val_score(model, X_train1, y_train1, cv=3, scoring='roc_auc')
        return scores.mean()
    optimizer1 = BayesianOptimization(f=xgb_cv_model1, pbounds=pbounds, random_state=42)
    optimizer1.maximize(init_points=5, n_iter=25)
    best_params_model1 = optimizer1.max['params']
    best_params_model1['n_estimators'] = int(best_params_model1['n_estimators'])
    model1 = XGBClassifier(**best_params_model1,scale_pos_weight=scale_pos_weight1,use_label_encoder=False,eval_metric='logloss',random_state=42)
    model1.fit(X_train1, y_train1)
    def xgb_cv_model2(n_estimators, gamma, learning_rate, colsample_bytree,reg_lambda, reg_alpha, subsample, min_child_weight):
        model = XGBClassifier(n_estimators=int(n_estimators),gamma=gamma,
            learning_rate=learning_rate,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            min_child_weight=min_child_weight,
            scale_pos_weight=scale_pos_weight2,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42)
        scores = cross_val_score(model, X_train2, y_train2, cv=3, scoring='roc_auc')
        return scores.mean()
    optimizer2 = BayesianOptimization(f=xgb_cv_model2, pbounds=pbounds, random_state=42)
    optimizer2.maximize(init_points=5, n_iter=25)
    best_params_model2 = optimizer2.max['params']
    best_params_model2['n_estimators'] = int(best_params_model2['n_estimators'])
    model2 = XGBClassifier(**best_params_model2,scale_pos_weight=scale_pos_weight2,use_label_encoder=False,eval_metric='logloss',random_state=42)
    model2.fit(X_train2, y_train2)
    return {"model1": {"model": model1,"X_test": X_test1,"y_test": y_test1},"model2": {"model": model2,"X_test": X_test2,"y_test": y_test2}}
def xgb_regression_models(df):
    features_model1 = ['collateral_value', 'avg_balance_amt', 'missed_repayments','interest', 'tenure_years', 'vintage_in_months','number_of_loans', 'cheque_bounces', 'avg_rr_per_city']
    features_model2 = ['missed_repayments', 'cheque_bounces','number_of_loans', 'vintage_in_months']
    X1_train, X1_test, y1_train, y1_test = train_test_split(df[features_model1],  df['RR'], test_size=0.2, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(df[features_model2],  df['RR'], test_size=0.2, random_state=42)
    pbounds = {'n_estimators': (100, 300),'gamma': (1, 10),'learning_rate': (0.01, 0.1),'colsample_bytree': (0.5, 0.9),'reg_lambda': (1, 10),'reg_alpha': (1, 10),'subsample': (0.5, 0.8),'min_child_weight': (5, 20)}
    def xgb_cv_model1(n_estimators, gamma, learning_rate, colsample_bytree,
                      reg_lambda, reg_alpha, subsample, min_child_weight):
        model = XGBRegressor(n_estimators=int(n_estimators),gamma=gamma,learning_rate=learning_rate,colsample_bytree=colsample_bytree,reg_lambda=reg_lambda,reg_alpha=reg_alpha,
            subsample=subsample,min_child_weight=min_child_weight,random_state=42)
        scores = cross_val_score(model, X1_train, y1_train, cv=3, scoring='neg_root_mean_squared_error')
        return scores.mean()
    def xgb_cv_model2(n_estimators, gamma, learning_rate, colsample_bytree,reg_lambda, reg_alpha, subsample, min_child_weight):
        model = XGBRegressor(n_estimators=int(n_estimators),gamma=gamma,learning_rate=learning_rate,colsample_bytree=colsample_bytree,reg_lambda=reg_lambda,reg_alpha=reg_alpha,subsample=subsample,
            min_child_weight=min_child_weight,
            random_state=42)
        scores = cross_val_score(model, X2_train, y2_train, cv=3, scoring='neg_root_mean_squared_error')
        return scores.mean()
    optimizer1 = BayesianOptimization(f=xgb_cv_model1, pbounds=pbounds, random_state=42)
    optimizer1.maximize(init_points=5, n_iter=25)
    optimizer2 = BayesianOptimization(f=xgb_cv_model2, pbounds=pbounds, random_state=42)
    optimizer2.maximize(init_points=5, n_iter=25)
    best_params_model1 = optimizer1.max['params']
    best_params_model1['n_estimators'] = int(best_params_model1['n_estimators'])
    best_params_model2 = optimizer2.max['params']
    best_params_model2['n_estimators'] = int(best_params_model2['n_estimators'])
    model1 = XGBRegressor(**best_params_model1, random_state=42)
    model1.fit(X1_train, y1_train)
    model2 = XGBRegressor(**best_params_model2, random_state=42)
    model2.fit(X2_train, y2_train)
    return {"model1": {"model": model1, "X_train": X1_train, "y_train": y1_train, "X_test": X1_test, "y_test": y1_test},
        "model2": {"model": model2, "X_train": X2_train, "y_train": y2_train, "X_test": X2_test, "y_test": y2_test}}
def train_lgd_nn_models(df, epochs=200):
    def prepare_data(feature_cols, target_col):
        y = df[target_col].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_cols].copy())
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y.values, test_size=0.2, random_state=42)
        return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
            torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32),)
    class Net(nn.Module):
        def __init__(self, input_dim, use_sigmoid=True):
            super(Net, self).__init__()
            layers = [
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)]
            if use_sigmoid:
                layers.append(nn.Sigmoid())
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            return self.model(x)
    features_rr = ['collateral_value', 'avg_balance_amt', 'missed_repayments','interest', 'tenure_years', 'vintage_in_months','number_of_loans', 'cheque_bounces', 'avg_rr_per_city']
    features_pc = ['missed_repayments', 'cheque_bounces','number_of_loans', 'vintage_in_months']
    X_rr_train, X_rr_test, y_rr_train, y_rr_test = prepare_data(features_rr, 'RR')
    X_pc_train, X_pc_test, y_pc_train, y_pc_test = prepare_data(features_pc, 'PC')
    y_lgd = df['lgd'].values
    _, _, y_lgd_train_np, y_lgd_test_np = train_test_split(
        df[features_rr].copy(), y_lgd, test_size=0.2, random_state=42)
    y_lgd_train = torch.tensor(y_lgd_train_np.reshape(-1, 1), dtype=torch.float32)
    y_lgd_test = torch.tensor(y_lgd_test_np.reshape(-1, 1), dtype=torch.float32)
    rr_model = Net(X_rr_train.shape[1], use_sigmoid=False)
    pc_model = Net(X_pc_train.shape[1], use_sigmoid=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(rr_model.parameters()) + list(pc_model.parameters()), lr=0.001)
    loss_history = []
    for epoch in range(epochs):
        rr_out = rr_model(X_rr_train)
        pc_out = pc_model(X_pc_train)
        lgd_pred = (1 - rr_out) * (1 - pc_out)
        loss = criterion(lgd_pred, y_lgd_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            rr_pred_test = rr_model(X_rr_test)
            pc_pred_test = pc_model(X_pc_test)
            lgd_test_pred = (1 - rr_pred_test) * (1 - pc_pred_test)
            test_loss = criterion(lgd_test_pred, y_lgd_test)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
        loss_history.append(loss.item())
    rr_pred_test = rr_model(X_rr_test)
    pc_pred_test = pc_model(X_pc_test)
    lgd_test_pred = (1 - rr_pred_test) * (1 - pc_pred_test)
    return {"rr_model": rr_model,"pc_model": pc_model,"loss_history": loss_history,"y_test": y_lgd_test.detach().numpy(),"y_pred": lgd_test_pred.detach().numpy()}


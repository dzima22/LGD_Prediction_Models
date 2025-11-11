from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import os

def evaluate_logit_models(results, threshold=0.5):
    metrics = {}
    for name, data in results.items():
        model = data["model"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        y_pred_prob = model.predict(X_test)
        y_pred_class = (y_pred_prob > threshold).astype(int)
        auc = roc_auc_score(y_test, y_pred_prob)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        aic = model.aic
        bic = model.bic
        acc = accuracy_score(y_test, y_pred_class)
        cm = confusion_matrix(y_test, y_pred_class)
        report = classification_report(
            y_test,
            y_pred_class,
            target_names=["Nie wyleczeni", "Wyleczeni"],
            digits=4,
            output_dict=True)
        metrics[name] = {"AIC": round(aic, 2),
            "BIC": round(bic, 2),
            "AUC": round(auc, 3),
            "Accuracy": round(acc, 4),
            "Confusion Matrix": cm,
            "ROC": {"fpr": fpr, "tpr": tpr},
            "Classification Report": report}
    return metrics
def evaluate_logit_adasyn(results, threshold=0.8074):
    model = results["model"]
    X_test = results["X_test"]
    y_test = results["y_test"]
    y_pred_prob = model.predict(X_test)
    y_pred_class = (y_pred_prob >= threshold).astype(int)
    auc = roc_auc_score(y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    return {"y_test": y_test,"y_pred_prob": y_pred_prob,"y_pred_class": y_pred_class,"auc": auc,"fpr": fpr,"tpr": tpr}
def evaluate_xgb_models(results):
    metrics = {}
    for name in ["model1", "model2"]:
        model = results[name]["model"]
        X_test = results[name]["X_test"]
        y_test = results[name]["y_test"]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        metrics[name] = {"AUC": auc,"Confusion Matrix": cm,"Accuracy": acc,"Precision": prec,"Recall": rec,"F1": f1,"fpr": fpr,"tpr": tpr,"y_test": y_test,"y_pred": y_pred,"y_proba": y_proba}
    return metrics
def evaluate_xgb_regression(results):
    metrics = {}
    for name in ["model1", "model2"]:
        model = results[name]["model"]
        X_test = results[name]["X_test"]
        y_test = results[name]["y_test"]
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics[name] = {"RMSE": rmse,"MAE": mae,"R2": r2,"y_test": y_test,"y_pred": y_pred}
    return metrics
def evaluate_weighted_rr(results, df):
    weighted_results = []
    for name in ["model1", "model2"]:
        model = results[name]["model"]
        X_test = results[name]["X_test"]
        y_test = results[name]["y_test"]
        y_pred = model.predict(X_test)
        subset = X_test.copy()
        subset["RR_real"] = y_test
        subset["RR_pred"] = y_pred
        subset["loan_amount"] = df.loc[y_test.index, "loan_amount"]
        real_weighted_rr = np.average(subset["RR_real"], weights=subset["loan_amount"])
        pred_weighted_rr = np.average(subset["RR_pred"], weights=subset["loan_amount"])
        weighted_results.append({
            "Model": f"XGB {name.upper()}",
            "Real Weighted RR": round(real_weighted_rr, 4),
            "Pred Weighted RR": round(pred_weighted_rr, 4)})
    return pd.DataFrame(weighted_results)
def save_output(content, filename, folder="visuals_folder"):
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)
    if hasattr(content, "summary"):
        text = content.summary().as_text()
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
    elif isinstance(content, str):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
    elif hasattr(content, "to_csv"):
        content.to_csv(save_path, index=False)
    else:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(str(content))



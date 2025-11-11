import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score,f1_score,precision_score,recall_score,auc
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

def top_cities(df,save_folder="visuals_folder"):
    city_counts = df['city'].value_counts().head(10)
    os.makedirs(save_folder, exist_ok=True)
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 7))
    bars = sns.barplot(x=city_counts.values, y=city_counts.index, palette="magma")
    for index, value in enumerate(city_counts.values):
        plt.text(value + 5, index, str(value), va='center', fontweight='bold', fontsize=10)
    plt.title("Top 10 miast z największą liczbą klientów", fontsize=14, fontweight='bold')
    plt.xlabel("Liczba klientów", fontsize=12)
    plt.ylabel("")  
    plt.xlim(0, max(city_counts.values) + 50)
    plt.tight_layout()
    save_path = os.path.join(save_folder, "top_cities.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
def loan_type_distribution(df,save_folder="visuals_folder"):
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#59a14f']
    os.makedirs(save_folder, exist_ok=True)
    mapping = {
        'Two-Wheeler': 'Motocykl',
        'Car': 'Samochód',
        'Personal': 'Pożyczka osobista',
        'Consumer-Durable': 'Konsumpcja'}
    translated = df['loan_type'].map(mapping).fillna(df['loan_type'])
    loan_type_counts = translated.value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        loan_type_counts,
        labels=loan_type_counts.index,
        autopct='%1.1f%%',
        wedgeprops=dict(width=0.4),
        pctdistance=0.81,
        startangle=90,
        colors=colors,
        textprops={'fontsize': 12, 'color': 'black'}
    )
    plt.text(0, -1.2, 'Typy zobowiązania', ha='center', fontsize=12, fontweight='bold')
    ax.axis('equal')
    plt.savefig(os.path.join(save_folder, "loan_type_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
def heatmap(df,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    mapping = {
    'loan_amount': 'Kwota kredytu',
    'collateral_value': 'Wartość zabezpieczenia',
    'cheque_bounces': 'Zwrócone czeki',
    'number_of_loans': 'Liczba zobowiązań',
    'missed_repayments': 'Zaległe raty',
    'vintage_in_months': 'Długość relacji z bankiem',
    'tenure_years': 'Okres spłaty',
    'interest': 'Oprocentowanie',
    'monthly_emi': 'Miesięczna rata',
    'year': 'Rok udzielenia',
    'total_repayment': 'Całkowita spłata',
    'avg_balance_amt': 'Uśredniona wartość salda'}
    plt.figure(figsize=(12, 9))
    sns.heatmap(df[['loan_amount','collateral_value','cheque_bounces','number_of_loans','missed_repayments','vintage_in_months','tenure_years','interest','monthly_emi','year','total_repayment','avg_balance_amt']].rename(columns=mapping).corr(), annot=True, fmt=".2f", cmap='viridis', square=True, cbar=True)
    plt.title('', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
def average_account_balance(df,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(data= df.groupby('lgd_category')['avg_balance_amt'].median().reset_index(),x='lgd_category',y='avg_balance_amt',palette='Set2',width=0.6)
    for i in ax.patches:
        ax.text(i.get_x() + i.get_width() / 2,i.get_height() + 200,f"{i.get_height():,.2f}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    ax.set_xlabel("", fontsize=11)
    ax.set_ylabel("", fontsize=11)
    ax.set_title("", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "average_account_balance.png"), dpi=300, bbox_inches='tight')
    plt.close()
def two_distributions(df,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    wedges1, texts1, autotexts1 = axs[0].pie(df['lgd_category'].value_counts(),
    labels=df['lgd_category'].value_counts().index,
    startangle=90,
    autopct='%1.1f%%',
    colors=['#55a868', '#dd8452', '#c44e52', '#4c72b0','#9370DB'] ,
    wedgeprops=dict(width=0.4),
    pctdistance=0.81,
    textprops=dict(color="black", fontsize=12))
    axs[0].text(0, -1.4, 'Klasyfikacja strat LGD według wzoru(18)', ha='center', fontsize=13, fontweight='bold')
    wedges2, texts2, autotexts2 = axs[1].pie(df['lgd_category_PC_RR_approach'].value_counts(),
    labels=df['lgd_category_PC_RR_approach'].value_counts().index,
    startangle=90,
    autopct='%1.1f%%',
    colors=['#55a868', '#dd8452', '#c44e52', '#4c72b0'] ,
    wedgeprops=dict(width=0.4),
    pctdistance=0.81,
    textprops=dict(color="black", fontsize=12))
    axs[1].text(0, -1.4, 'Klasyfikacja strat LGD według wzoru(4)', ha='center', fontsize=13, fontweight='bold')
    for i in autotexts1 + autotexts2:
        i.set_color('black')
        i.set_fontsize(12)
        i.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "two_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
def roc_curves(metrics,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for i, (name, data) in enumerate(metrics.items(), 1):
        plt.subplot(1, len(metrics), i)
        plt.plot(data["ROC"]["fpr"], data["ROC"]["tpr"], label=f'{name} (AUC={data["AUC"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Losowy klasyfikator')
        plt.title(f'Model: {name.upper()}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "roc_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
def confusion_matrices(metrics,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    def plot_conf_matrix(cm, title, subplot_pos):
        plt.subplot(1, 2, subplot_pos)
        plt.imshow(cm, cmap="YlOrRd")
        plt.title(title)
        plt.xlabel("Predykcja")
        plt.ylabel("Rzeczywiste")
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                color = "white" if value > cm.max() * 0.5 else "black"
                plt.text(j, i, str(value), ha="center", va="center", 
                         color=color, fontsize=13, fontweight='bold')
        plt.grid(False)
    plt.figure(figsize=(14, 6))
    plot_conf_matrix(metrics["model1"]["Confusion Matrix"], "Model 1 — Macierz pomyłek", 1)
    plot_conf_matrix(metrics["model2"]["Confusion Matrix"], "Model 2 — Macierz pomyłek", 2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "confusion_matrices.png"), dpi=300, bbox_inches='tight')
    plt.close()
def roc_adasyn(metrics,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(metrics["fpr"], metrics["tpr"], label=f'AUC = {metrics["auc"]:.2f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Losowy klasyfikator')
    plt.title('Krzywa ROC (Model z ADASYN)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "roc_adasyn.png"), dpi=300, bbox_inches='tight')
    plt.close()
def confusion_matrix_adasyn(metrics,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    cm = confusion_matrix( metrics["y_test"], metrics["y_pred_class"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="YlOrRd")
    plt.title("Macierz pomyłek (Model z ADASYN)")
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywiste")
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "white" if value > cm.max() * 0.5 else "black"
            plt.text(j, i, str(value), ha="center", va="center",
                     color=color, fontsize=13, fontweight='bold')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "confusion_matrix_adasyn.png"), dpi=300, bbox_inches='tight')
    plt.close()
def xgb_roc_curves(metrics,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for i, name in enumerate(["model1", "model2"], 1):
        plt.subplot(1, 2, i)
        plt.plot(metrics[name]["fpr"], metrics[name]["tpr"], label=f'{name} (AUC = {metrics[name]["AUC"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'Model: {name.upper()}')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "xgb_roc_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
def xgb_comparison_roc(metrics,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    fpr1, tpr1, _ = roc_curve(metrics["model1"]["y_test"], metrics["model1"]["y_proba"])
    fpr2, tpr2, _ = roc_curve(metrics["model2"]["y_test"], metrics["model2"]["y_proba"])
    plt.figure(figsize=(10, 5))
    plt.plot(fpr1, tpr1, color='blue', lw=2, label=f"Model 1 (AUC = { auc(fpr1, tpr1):.2f})")
    plt.plot(fpr2, tpr2, color='green', lw=2, label=f"Model 2 (AUC = {auc(fpr2, tpr2):.2f})")
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Porównanie krzywych ROC – Modele XGBoost")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, "xgb_comparison_roc.png"), dpi=300, bbox_inches='tight')
    plt.close()
def xgb_conf_matrices(metrics,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    def plot_conf_matrix(cm, title, subplot_pos):
        plt.subplot(1, 2, subplot_pos)
        plt.imshow(cm, cmap="YlOrRd")
        plt.title(title)
        plt.xlabel("Predykcja")
        plt.ylabel("Rzeczywiste")
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                color = "white" if value > cm.max() * 0.5 else "black"
                plt.text(j, i, str(value), ha="center", va="center",
                         color=color, fontsize=13, fontweight='bold')
        plt.grid(False)
    plt.figure(figsize=(14, 6))
    plot_conf_matrix(metrics["model1"]["Confusion Matrix"], "Model 1 — Macierz pomyłek", 1)
    plot_conf_matrix(metrics["model2"]["Confusion Matrix"], "Model 2 — Macierz pomyłek", 2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "xgb_conf_matrice.png"), dpi=300, bbox_inches='tight')
    plt.close()
def xgb_regression_results(metrics,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(metrics["model1"]["y_test"], metrics["model1"]["y_pred"], alpha=0.5, color="steelblue", edgecolor='k')
    plt.plot([0, 1.1], [0, 1.1], '--', color='red')
    plt.xlabel("Wartość rzeczywista (RR)")
    plt.ylabel("Wartość przewidywana (RR)")
    plt.title("Model 1 — XGB Regressor")
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.scatter(metrics["model2"]["y_test"], metrics["model2"]["y_pred"], alpha=0.5, color="darkorange", edgecolor='k')
    plt.plot([0, 1.1], [0, 1.1], '--', color='red')
    plt.xlabel("Wartość rzeczywista (RR)")
    plt.ylabel("Wartość przewidywana (RR)")
    plt.title("Model 2 — XGB Regressor")
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "xgb_regression_results.png"), dpi=300, bbox_inches='tight')
    plt.close()
def lgd_loss_curves(loss_long, loss_short,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    epochs = [10 * i for i in range(1, len(loss_long) + 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_long, marker='o', color='#001F66', label='Zestaw 9 zmiennych')
    plt.plot(epochs, loss_short, marker='o', color='#FFD966', label='Zestaw 4 zmiennych')
    plt.xlabel("Iteracja (Epoka)")
    plt.ylabel("Średni błąd kwadratowy (MSE)")
    plt.title("Porównanie krzywych strat sieci neuronowych LGD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "lgd_loss_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
def lgd_pred_vs_real(results_long, results_short,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    axs[0].scatter(np.clip(results_long["y_test"], 0, 1), np.clip(results_long["y_pred"], 0, 1), alpha=0.6, color='royalblue', label='Predykcje')
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='red', label='Idealne dopasowanie')
    axs[0].set_title('Zestaw 9 zmiennych')
    axs[0].set_xlabel('Wartość rzeczywista')
    axs[0].set_ylabel('Wartość prognozowana')
    axs[0].legend()
    axs[0].grid(True)
    axs[1].scatter(np.clip(results_short["y_test"], 0, 1), np.clip(results_short["y_pred"], 0, 1), alpha=0.6, color='#FFD966', label='Predykcje')
    axs[1].plot([0, 1], [0, 1], linestyle='--', color='red', label='Idealne dopasowanie')
    axs[1].set_title('Zestaw 4 zmiennych')
    axs[1].set_xlabel('Wartość rzeczywista')
    axs[1].set_ylabel('Wartość prognozowana')
    axs[1].legend()
    axs[1].grid(True)
    plt.suptitle('Porównanie rzeczywistych i przewidywanych wartości LGD', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_folder, "lgd_pred_vs_real.png"), dpi=300, bbox_inches='tight')
    plt.close()
def lgd_error_by_year(df, results_long, results_short,save_folder="visuals_folder"):
    os.makedirs(save_folder, exist_ok=True)
    _, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_test['default_date'] = pd.to_datetime(df_test['default_date'])
    df_error_long = pd.DataFrame({'default_date': df_test['default_date'].values,'lgd_error': np.abs(results_long["y_test"].flatten() - results_long["y_pred"].flatten())})
    df_error_short = pd.DataFrame({'default_date': df_test['default_date'].values,'lgd_error': np.abs(results_short["y_test"].flatten() - results_short["y_pred"].flatten())})
    df_error_long['year'] = df_error_long['default_date'].dt.year
    df_error_short['year'] = df_error_short['default_date'].dt.year
    plt.figure(figsize=(10, 5))
    plt.plot(df_error_long.groupby('year')['lgd_error'].mean().index.astype(str), df_error_long.groupby('year')['lgd_error'].mean().values, label='Zestaw 9 zmiennych', color='royalblue', marker='o')
    plt.plot(df_error_short.groupby('year')['lgd_error'].mean().index.astype(str), df_error_short.groupby('year')['lgd_error'].mean().values, label='Zestaw 4 zmiennych', color='#FFD966', marker='o')
    plt.title("Średni błąd bezwzględny (MAE) modeli LGD w podziale na lata")
    plt.xlabel("Rok")
    plt.ylabel("Średni błąd bezwzględny")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "lgd_error_by_year.png"), dpi=300, bbox_inches='tight')
    plt.close()

import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from modules.config import (numeric_features)

def hellwig_selection(df, target_col, min_k=2, max_k=5):
    """
    Metoda Hellwiga dla 12 wybranych zmiennych liczbowych.
    """
    X = df[numeric_features].copy()
    corr_with_y = X.apply(lambda col: df[target_col].corr(col))
    def hellwig_score(subset):
        r_yi_sq = np.array([corr_with_y[feature] ** 2 for feature in subset])
        r_ij_sq_sum = np.array([sum( X.corr().loc[feature] ** 2) for feature in subset])
        return sum(r_yi_sq / r_ij_sq_sum)
    scores = []
    best_subset, best_score = None, -np.inf
    for k in range(min_k, max_k + 1):
        for subset in itertools.combinations(X.columns, k):
            score = hellwig_score(subset)
            scores.append((subset, score))
            if score > best_score:
                best_score, best_subset = score, subset
    results = pd.DataFrame(scores, columns=['subset', 'hellwig_score']).sort_values(by='hellwig_score', ascending=False)
    return results, best_subset
def vif_selection(df):
    X = sm.add_constant(df[numeric_features].copy())
    vif_data = pd.DataFrame({
        'Variable': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
    return vif_data
def forward_selection(df, target_col, threshold_in=0.05):
    X = df[numeric_features].copy()
    included = []
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_col in excluded:
            model = sm.OLS(df[target_col], sm.add_constant(X[included + [new_col]])).fit()
            new_pval[new_col] = model.pvalues[new_col]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
        if not changed:
            break
    return included



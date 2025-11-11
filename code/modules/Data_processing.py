# %%
import pandas as pd
import numpy as np
import re

def load_data(loans_path, balance_path, repayment_path):
    loans = pd.read_csv(loans_path)
    balance = pd.read_csv(balance_path)
    repayment = pd.read_csv(repayment_path)
    return loans, balance, repayment

def preprocess_loans(loans):
    loans['disbursal_date'] = pd.to_datetime(loans['disbursal_date'])
    loans['default_date'] = pd.to_datetime(loans['default_date'])
    loans['year'] = loans['default_date'].dt.year
    loans['quarter'] = loans['default_date'].dt.quarter.apply(lambda x: f'Q{x}')
    loans['month'] = loans['default_date'].dt.month_name()
    loans = loans[loans['default_date'] <= '2024-12-31']
    return loans

def merge_data(loans, balance, repayment):
    avg_bal = (balance.groupby('loan_acc_num')['balance_amount'].median().reset_index().rename(columns={'balance_amount': 'avg_balance_amt'}))
    sum_repayment = (repayment.groupby('loan_acc_num')['repayment_amount'].sum().reset_index().rename(columns={'repayment_amount': 'total_repayment'}))
    df = loans.merge(sum_repayment, on='loan_acc_num', how='inner')
    df = df.merge(avg_bal, on='loan_acc_num', how='inner')
    return df

def categorize_lgd(lgd):
    if lgd < 0:
        return 'Odzysk większy niż ekspozycja'
    elif lgd <= 0.2:
        return 'Niska strata'
    elif lgd <= 0.4:
        return 'Umiarkowana strata'
    elif lgd <= 0.6:
        return 'Wysoka strata'
    else:
        return 'Bardzo wysoka strata'

def compute_lgd(df):
    df['lgd'] = (df['loan_amount'] - df['collateral_value'] - df['total_repayment']) / df['loan_amount']
    df['lgd_category'] = df['lgd'].apply(categorize_lgd)
    return df

def extract_city(address_series):
    city = []
    for i in address_series:
        try:
            part = (i.replace('\n', ',')).split(',')[2]
            cleaned = re.sub(r'[-\d+]', '', part).strip().title()
            city.append(" ".join(cleaned.split()))
        except:
            city.append("Unknown")
    return city

def add_city_column(df):
    city_list = extract_city(df['customer_address'])
    df.insert(3, 'city', city_list)
    return df

def basic_statistics(df):
    numeric_df = df[['loan_amount', 'collateral_value', 'cheque_bounces', 'number_of_loans', 'missed_repayments']]
    statistics = pd.DataFrame({'Średnia': numeric_df.mean(), 
                               'Mediana': numeric_df.median(),
                               'Odchylenie standardowe': numeric_df.std(),
                               'Wariancja': numeric_df.var(),
                                'Skośność': numeric_df.skew(),
                                'Kurtoza': numeric_df.kurt()}).round(2)
    return statistics

def get_value_counts(df):
    categorical_columns = df.select_dtypes(include='O').columns
    for col in categorical_columns[3:]:
        print(f'\nValue counts for {col}:')
        print(df[col].value_counts())

def calculate_lgd_second_approach(df):
    #Refunded amount
    df['recovered_total'] = df['total_repayment'] + df['collateral_value']

    #Loan been repaid in full
    df['cured'] = df['recovered_total'] >= df['loan_amount']
    df["PC"] = df["cured"].astype(float)

    #Recovery Rate
    df['RR'] = ((df['collateral_value'] + df['total_repayment']) / df['loan_amount']).round(4)

    #LGD calculated based on RR and Cured approach
    df['LGD_PC_RR_approach'] = np.where(df['cured'], 0, 1 - df['RR'])

    #LGD categories based on RR and PC approach
    df['lgd_category_PC_RR_approach'] = df['LGD_PC_RR_approach'].apply(categorize_lgd)

    #Average RR per city
    rr_per_city = (df.groupby('city', as_index=False)['RR'].mean().rename(columns={'RR': 'avg_rr_per_city'}))

    df = df.merge(rr_per_city, on='city', how='left')

    return df



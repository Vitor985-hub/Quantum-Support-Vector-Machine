from sklearn.datasets import load_breast_cancer
import pandas as pd 

def load_dataset():
    """Carrega o Breast Cancer dataset e retorna um DataFrame"""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def normalize_data(df, feature_col):
    # Normaliza os dados para [0,1] usando MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_col]), columns=feature_col)
    return df_scaled, scaler

def apply_pca(df_scaled, n_components=2):
    # Reduz dimensionalidade com PCA
    pca = PCA(n_components=n_components)
    df_pca = pd.DataFrame(pca.fit_transform(df_scaled), columns=[f"PC{i+1}" for i in range(n_components)])
    return df_pca, pca
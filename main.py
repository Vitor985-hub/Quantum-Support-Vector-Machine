import dataset_loader
import preprocessing

df, data = dataset_loader.load_dataset()
feature_cols = df.columns[:-1]

df_scaled, scaler = preprocessing.normalize_data(df, feature_cols)
print('primeiras linhas normalizadas')
print(df_scaled.head())

df_pca, pca = preprocessing.apply_pca(df_scaled, n_components=2)
print('\n primeiras linhas após pca: ')
print(df_pca.head())

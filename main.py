import dataset_loader
import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

df, data = dataset_loader.load_dataset()
feature_cols = df.columns[:-1]

df_scaled, scaler = preprocessing.normalize_data(df, feature_cols)
print('primeiras linhas normalizadas')
print(df_scaled.head())

df_pca, pca = preprocessing.apply_pca(df_scaled, n_components=2)
print('\n primeiras linhas após pca: ')
print(df_pca.head())

# Scatter plot PCA
def plot_pca_scatter(df_pca, df_target):
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x="PC1", y="PC2",
        hue=df_target,
        palette={0: "red", 1: "green"},
        data=df_pca
    )
    plt.title("PCA - Breast Cancer Dataset")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(title="Target", labels=["Maligno (0)", "Benigno (1)"])
    plt.show()

# Chamando a função
plot_pca_scatter(df_pca, df['target'])
import dataset_loader

df, data = dataset_loader.load_dataset()

print('shape do dataset', df.shape)
print('colunas', df.columns.to_list())
print('primeiras linhas do dataset')
print(df.head())
print(df.describe())
print(df['target'].value_counts())

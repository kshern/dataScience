import pandas as pd

# 加载数据
train_data = pd.read_csv('data/processed_train.csv')
print("训练集列名:", train_data.columns.tolist())

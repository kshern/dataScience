import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def process_data(df):
    # 复制数据以避免修改原始数据
    data = df.copy()
    
    # 处理年龄缺失值
    age_mean = data['Age'].mean()
    data['Age'].fillna(age_mean, inplace=True)
    
    # 处理Embarked缺失值
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    # 处理Fare缺失值
    fare_mean = data['Fare'].mean()
    data['Fare'].fillna(fare_mean, inplace=True)
    
    # 创建家庭规模特征
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    # 创建是否单独旅行的特征
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # 对分类特征进行编码
    le = LabelEncoder()
    data['Sex'] = le.fit_transform(data['Sex'])
    data['Embarked'] = le.fit_transform(data['Embarked'])
    
    # 特征缩放
    data['Age'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()
    data['Fare'] = (data['Fare'] - data['Fare'].mean()) / data['Fare'].std()
    
    # 选择最终特征
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                'FamilySize', 'IsAlone']
    
    return data[features]

# 读取原始数据
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# 处理训练集
processed_train = process_data(train_data)
if 'Survived' in train_data.columns:
    processed_train['Survived'] = train_data['Survived']

# 处理测试集
processed_test = process_data(test_data)

# 保存处理后的数据
processed_train.to_csv('data/processed_train_no_title.csv', index=False)
processed_test.to_csv('data/processed_test_no_title.csv', index=False)

print("数据处理完成！")
print(f"训练集形状: {processed_train.shape}")
print(f"测试集形状: {processed_test.shape}")
print("\n特征列表:")
print(processed_train.columns.tolist())

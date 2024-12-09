import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 生成示例数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 初始化K折交叉验证
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# 初始化模型
model = LogisticRegression()

# 存储每折的准确率
fold_accuracies = []

# 创建一个数组来追踪每个样本被用作验证集的次数
validation_count = np.zeros(len(X))

# 执行K折交叉验证
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    # 记录本折中用作验证集的样本
    validation_count[val_idx] += 1
    
    # 分割训练集和验证集
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # 打印每折的数据分布情况
    print(f'\nFold {fold}:')
    print(f'训练集样本索引（前5个）: {train_idx[:5]}...')
    print(f'验证集样本索引（前5个）: {val_idx[:5]}...')
    print(f'训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}')
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 在验证集上预测
    y_pred = model.predict(X_val)
    
    # 计算准确率
    accuracy = accuracy_score(y_val, y_pred)
    fold_accuracies.append(accuracy)
    
    print(f'Accuracy: {accuracy:.4f}')

# 验证每个样本是否只被用作验证集一次
print('\n验证集使用情况:')
print(f'每个样本被用作验证集的次数的唯一值: {np.unique(validation_count)}')
print(f'确认每个样本只被用作验证集一次: {np.all(validation_count == 1)}')

# 计算平均准确率
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f'\nCross-validation results:')
print(f'Mean accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def load_and_prepare_data():
    # 加载数据
    train_data = pd.read_csv('data/processed_train.csv')
    test_data = pd.read_csv('data/processed_test.csv')
    
    # 分离特征和目标变量
    X = train_data.drop('Survived', axis=1)
    y = train_data['Survived']
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    test_scaled = scaler.transform(test_data)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_val, y_train, y_val, test_scaled, scaler

def create_model(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, X_val, y_train, y_val):
    # 添加早停策略
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )
    
    return history

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失曲线
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 绘制准确率曲线
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def generate_predictions(model, test_scaled):
    # 生成预测结果
    predictions = model.predict(test_scaled)
    predictions = (predictions > 0.5).astype(int)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'PassengerId': range(892, 892 + len(predictions)),
        'Survived': predictions.flatten()
    })
    
    submission.to_csv('data/nn_submission.csv', index=False)
    print("预测结果已保存到 'nn_submission.csv'")

if __name__ == '__main__':
    # 准备数据
    X_train, X_val, y_train, y_val, test_scaled, scaler = load_and_prepare_data()
    
    # 创建模型
    model = create_model(input_shape=(X_train.shape[1],))
    
    # 训练模型
    history = train_model(model, X_train, X_val, y_train, y_val)
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 生成预测结果
    generate_predictions(model, test_scaled)

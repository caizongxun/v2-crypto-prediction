"""
模型融合系統 - 使用 Stacking 技术統合多個模型
目標: 達到 70%+ 準確率
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class ModelEnsemble:
    """
    模型融合系統 - Stacking 横梏
    
    思路:
    1. 第一层: 訓練基礎模型 (也樹模型、丹寶模型等)
    2. 第二層: 使用基礎模型的預測作為新特徵
    3. 第三層: 訓練最終meta模型進行統合
    """
    
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.validation_scores = {}
        
    def create_base_models(self):
        """建立基礎模型"""
        print("\n" + "="*60)
        print("模型融合 - 創建基礎模型")
        print("="*60)
        
        self.base_models = {
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=11,
                num_leaves=63,
                learning_rate=0.05,
                reg_alpha=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=9,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=0
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'adaboost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.05,
                random_state=42
            )
        }
        
        print(f"\n已創建 {len(self.base_models)} 個基礎模型:")
        for name in self.base_models.keys():
            print(f"  - {name.upper()}")
    
    def train_base_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """訓練基礎模型並計算驗證集準確率"""
        print("\n" + "="*60)
        print("訓練基礎模型")
        print("="*60)
        
        # 正規化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 訓練每個基礎模型
        self.train_preds = np.zeros((len(X_train), len(self.base_models)))
        self.val_preds = np.zeros((len(X_val), len(self.base_models)))
        self.test_preds = np.zeros((len(X_test), len(self.base_models)))
        
        for idx, (name, model) in enumerate(self.base_models.items()):
            print(f"\n訓練 {name.upper()}...")
            
            # 訓練模型
            model.fit(X_train_scaled, y_train)
            
            # 獲取驗證集預測
            val_pred = model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, val_pred)
            self.validation_scores[name] = val_accuracy
            
            print(f"  驗證集準確率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            
            # 獲取訓練、驗證、測試上的統計数字
            self.train_preds[:, idx] = model.predict(X_train_scaled)
            self.val_preds[:, idx] = val_pred
            self.test_preds[:, idx] = model.predict(X_test_scaled)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_meta_model(self, X_train_meta, y_train, X_val_meta, y_val, X_test_meta, y_test):
        """訓練 Meta 模型進行最終融合"""
        print("\n" + "="*60)
        print("訓練 Meta 模型 (最終融合層)")
        print("="*60)
        
        # Meta 模型使用隱藏 Logistic Regression
        self.meta_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        # 訓練
        self.meta_model.fit(X_train_meta, y_train)
        
        # 驗證
        val_pred = self.meta_model.predict(X_val_meta)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        # 測試
        test_pred = self.meta_model.predict(X_test_meta)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"\nMeta 模型驗證集準確率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"Meta 模型測試集準確率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        return test_pred, test_accuracy
    
    def fit_ensemble(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """訓練整個融合模型"""
        # 創建基礎模型
        self.create_base_models()
        
        # 訓練基礎模型
        X_train_scaled, X_val_scaled, X_test_scaled = self.train_base_models(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # 訓練 Meta 模型
        test_pred, test_accuracy = self.train_meta_model(
            self.train_preds, y_train,
            self.val_preds, y_val,
            self.test_preds, y_test
        )
        
        return test_pred, test_accuracy
    
    def print_summary(self, accuracy):
        """打印訓練結果汇总"""
        print("\n" + "="*60)
        print("基礎模型驗證集結果")
        print("="*60)
        
        for name, score in sorted(self.validation_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{name.upper():20s}: {score:.4f} ({score*100:.2f}%)")
        
        print("\n" + "="*60)
        print(f"最終融合準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"比較基線 (51.79%): {(accuracy-0.5179)*100:+.2f}%")
        print("="*60)


if __name__ == '__main__':
    from data import load_btc_data
    from indicators import IndicatorCalculator
    from train_models_v2 import ModelTrainerV2
    from config import HF_TOKEN
    
    print("模型融合實驗 - 目標 70%+ 準確率")
    
    # 1. 載入数據
    print("\n[1/4] 載入数據...")
    df = load_btc_data(hf_token=HF_TOKEN)
    print(f"数據載入: {df.shape[0]} 根 K 線")
    
    # 2. 計算指標
    print("[2/4] 計算技術指標...")
    calc = IndicatorCalculator()
    indicators = calc.calculate_all(df)
    print(f"技術指標: {len(indicators)} 個")
    
    # 3. 構建特徵
    print("[3/4] 構建特徵...")
    trend_strength = np.ones(len(df)) * 0.5
    volatility_index = np.ones(len(df)) * 0.5
    direction_confirmation = np.ones(len(df)) * 0.5
    
    trainer = ModelTrainerV2(df, {})
    X, y = trainer.prepare_features_v2(
        indicators,
        {
            'trend_strength': trend_strength,
            'volatility_index': volatility_index,
            'direction_confirmation': direction_confirmation
        }
    )
    print(f"特徵数: {X.shape[1]}、敷數: {X.shape[0]}")
    
    # 4. 數據分割
    print("[4/4] 數據分割...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, shuffle=False
    )
    
    print(f"訓練集: {X_train.shape[0]}、驗證集: {X_val.shape[0]}、測試集: {X_test.shape[0]}")
    
    # 5. 訓練融合模型
    print("\n" + "#"*60)
    print("# 開始融合訓練")
    print("#"*60)
    
    ensemble = ModelEnsemble()
    test_pred, test_accuracy = ensemble.fit_ensemble(
        X_train, y_train['direction'].values,
        X_val, y_val['direction'].values,
        X_test, y_test['direction'].values
    )
    
    ensemble.print_summary(test_accuracy)
    
    # 收集細節指標
    print("\n" + "="*60)
    print("詳細準確率指標")
    print("="*60)
    print(f"Precision: {precision_score(y_test['direction'].values, test_pred):.4f}")
    print(f"Recall:    {recall_score(y_test['direction'].values, test_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test['direction'].values, test_pred):.4f}")

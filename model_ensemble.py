"""
模型融合系統 v2 - LightGBM + CatBoost 分開訓練，推理時融合
特點: 
  - 模型獨立訓練，易於偵錯和更新
  - 推理時同時導入兩個模型，加權融合
  - 支持保存/載入各模型
"""

import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


class DualModelEnsemble:
    """
    LightGBM + CatBoost 雙模型融合系統
    
    訓練階段:
      - 分開訓練 LightGBM 和 CatBoost
      - 各自驗證性能
      
    推理階段:
      - 同時導入兩個模型
      - 加權平均融合預測結果
    """
    
    def __init__(self, lightgbm_weight=0.5, catboost_weight=0.5):
        """
        初始化雙模型融合系統
        
        Args:
            lightgbm_weight: LightGBM 在融合中的權重 (0-1)
            catboost_weight: CatBoost 在融合中的權重 (0-1)
        """
        self.lightgbm_model = None
        self.catboost_model = None
        self.scaler = StandardScaler()
        
        # 權重和應為 1.0
        total = lightgbm_weight + catboost_weight
        self.lightgbm_weight = lightgbm_weight / total
        self.catboost_weight = catboost_weight / total
        
        self.validation_scores = {
            'lightgbm': {},
            'catboost': {}
        }
        
        self.model_dir = 'trained_models'
        os.makedirs(self.model_dir, exist_ok=True)
    
    def create_lightgbm_model(self):
        """建立 LightGBM 模型"""
        self.lightgbm_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=11,
            num_leaves=63,
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            is_unbalance=True
        )
        print("✓ LightGBM 模型已建立")
    
    def create_catboost_model(self):
        """建立 CatBoost 模型"""
        self.catboost_model = CatBoostClassifier(
            iterations=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0,
            loss_function='Logloss',
            eval_metric='Accuracy',
            od_type='Iter',
            od_wait=20
        )
        print("✓ CatBoost 模型已建立")
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """訓練 LightGBM 模型"""
        print("\n" + "="*60)
        print("訓練 LightGBM 模型")
        print("="*60)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 訓練
        self.lightgbm_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            eval_metric='binary_logloss',
            callbacks=[
                lgb.early_stopping(10, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # 驗證
        val_pred = self.lightgbm_model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        # 測試
        test_pred = self.lightgbm_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # 儲存結果
        self.validation_scores['lightgbm'] = {
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred)
        }
        
        print(f"驗證集準確率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"測試集準確率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Precision: {self.validation_scores['lightgbm']['val_precision']:.4f}")
        print(f"Recall: {self.validation_scores['lightgbm']['val_recall']:.4f}")
        print(f"F1 Score: {self.validation_scores['lightgbm']['val_f1']:.4f}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_catboost(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """訓練 CatBoost 模型"""
        print("\n" + "="*60)
        print("訓練 CatBoost 模型")
        print("="*60)
        
        # CatBoost 不需要特徵縮放，但可以使用
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 訓練
        self.catboost_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=0
        )
        
        # 驗證
        val_pred = self.catboost_model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        # 測試
        test_pred = self.catboost_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # 儲存結果
        self.validation_scores['catboost'] = {
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred)
        }
        
        print(f"驗證集準確率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"測試集準確率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Precision: {self.validation_scores['catboost']['val_precision']:.4f}")
        print(f"Recall: {self.validation_scores['catboost']['val_recall']:.4f}")
        print(f"F1 Score: {self.validation_scores['catboost']['val_f1']:.4f}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def predict_ensemble(self, X, use_probabilities=False):
        """
        使用融合模型進行預測
        
        Args:
            X: 輸入特徵 (np.ndarray)
            use_probabilities: 是否使用概率預測 (預測時更精確)
            
        Returns:
            融合預測結果 (0/1)
        """
        X_scaled = self.scaler.transform(X)
        
        if use_probabilities:
            # 使用概率預測，加權平均後取閾值 0.5
            lgb_proba = self.lightgbm_model.predict_proba(X_scaled)[:, 1]
            cb_proba = self.catboost_model.predict_proba(X_scaled)[:, 1]
            
            ensemble_proba = (self.lightgbm_weight * lgb_proba + 
                            self.catboost_weight * cb_proba)
            return (ensemble_proba >= 0.5).astype(int)
        else:
            # 使用硬預測，多數投票
            lgb_pred = self.lightgbm_model.predict(X_scaled)
            cb_pred = self.catboost_model.predict(X_scaled)
            
            # 加權投票
            votes = self.lightgbm_weight * lgb_pred + self.catboost_weight * cb_pred
            return (votes >= 0.5).astype(int)
    
    def predict_proba_ensemble(self, X):
        """
        獲取融合模型的預測概率
        
        Returns:
            融合預測概率 (0.0-1.0)
        """
        X_scaled = self.scaler.transform(X)
        
        lgb_proba = self.lightgbm_model.predict_proba(X_scaled)[:, 1]
        cb_proba = self.catboost_model.predict_proba(X_scaled)[:, 1]
        
        return (self.lightgbm_weight * lgb_proba + 
                self.catboost_weight * cb_proba)
    
    def fit_ensemble(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """訓練整個雙模型融合系統"""
        print("\n" + "#"*60)
        print("# LightGBM + CatBoost 雙模型融合訓練")
        print("#"*60)
        
        # 建立模型
        self.create_lightgbm_model()
        self.create_catboost_model()
        
        # 訓練 LightGBM
        X_train_scaled, X_val_scaled, X_test_scaled = self.train_lightgbm(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # 訓練 CatBoost
        self.train_catboost(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # 計算融合模型的性能
        print("\n" + "="*60)
        print("融合模型測試集性能")
        print("="*60)
        
        ensemble_pred = self.predict_ensemble(X_test, use_probabilities=True)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_precision = precision_score(y_test, ensemble_pred)
        ensemble_recall = recall_score(y_test, ensemble_pred)
        ensemble_f1 = f1_score(y_test, ensemble_pred)
        
        print(f"準確率: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
        print(f"Precision: {ensemble_precision:.4f}")
        print(f"Recall: {ensemble_recall:.4f}")
        print(f"F1 Score: {ensemble_f1:.4f}")
        
        return ensemble_pred, ensemble_accuracy
    
    def save_models(self, tag=''):
        """保存訓練好的模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{tag}" if tag else ""
        
        lgb_path = os.path.join(self.model_dir, f"lightgbm{tag}_{timestamp}.pkl")
        cb_path = os.path.join(self.model_dir, f"catboost{tag}_{timestamp}.pkl")
        scaler_path = os.path.join(self.model_dir, f"scaler{tag}_{timestamp}.pkl")
        
        joblib.dump(self.lightgbm_model, lgb_path)
        joblib.dump(self.catboost_model, cb_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\n✓ 模型已保存:")
        print(f"  - {lgb_path}")
        print(f"  - {cb_path}")
        print(f"  - {scaler_path}")
        
        return lgb_path, cb_path, scaler_path
    
    def load_models(self, lgb_path, cb_path, scaler_path):
        """載入訓練好的模型"""
        self.lightgbm_model = joblib.load(lgb_path)
        self.catboost_model = joblib.load(cb_path)
        self.scaler = joblib.load(scaler_path)
        
        print(f"\n✓ 模型已載入:")
        print(f"  - {lgb_path}")
        print(f"  - {cb_path}")
        print(f"  - {scaler_path}")
    
    def print_summary(self):
        """打印訓練結果總結"""
        print("\n" + "="*60)
        print("模型性能對比")
        print("="*60)
        
        print(f"\nLightGBM:")
        for metric, value in self.validation_scores['lightgbm'].items():
            print(f"  {metric:15s}: {value:.4f}")
        
        print(f"\nCatBoost:")
        for metric, value in self.validation_scores['catboost'].items():
            print(f"  {metric:15s}: {value:.4f}")
        
        print(f"\n融合權重:")
        print(f"  LightGBM: {self.lightgbm_weight*100:.1f}%")
        print(f"  CatBoost: {self.catboost_weight*100:.1f}%")


if __name__ == '__main__':
    from data import load_btc_data
    from indicators import IndicatorCalculator
    from train_models_v2 import ModelTrainerV2
    from config import HF_TOKEN
    
    print("LightGBM + CatBoost 雙模型融合 - 目標 70%+ 準確率")
    
    # 1. 載入數據
    print("\n[1/4] 載入數據...")
    df = load_btc_data(hf_token=HF_TOKEN)
    print(f"數據載入: {df.shape[0]} 根 K 線")
    
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
    print(f"特徵數: {X.shape[1]}、樣數: {X.shape[0]}")
    
    # 4. 數據分割
    print("[4/4] 數據分割...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, shuffle=False
    )
    
    print(f"訓練集: {X_train.shape[0]}、驗證集: {X_val.shape[0]}、測試集: {X_test.shape[0]}")
    
    # 5. 訓練雙模型融合系統
    print("\n" + "#"*60)
    print("# 開始訓練")
    print("#"*60)
    
    ensemble = DualModelEnsemble(lightgbm_weight=0.5, catboost_weight=0.5)
    ensemble_pred, ensemble_accuracy = ensemble.fit_ensemble(
        X_train, y_train['direction'].values,
        X_val, y_val['direction'].values,
        X_test, y_test['direction'].values
    )
    
    ensemble.print_summary()
    
    # 6. 保存模型
    lgb_path, cb_path, scaler_path = ensemble.save_models(tag='v1')
    
    # 7. 演示載入和推理
    print("\n" + "="*60)
    print("演示: 載入模型並進行推理")
    print("="*60)
    
    # 建立新實例並載入模型
    demo_ensemble = DualModelEnsemble()
    demo_ensemble.load_models(lgb_path, cb_path, scaler_path)
    
    # 在測試集上進行推理
    demo_pred = demo_ensemble.predict_ensemble(X_test.values, use_probabilities=True)
    demo_accuracy = accuracy_score(y_test['direction'].values, demo_pred)
    
    print(f"\n載入後推理準確率: {demo_accuracy:.4f} ({demo_accuracy*100:.2f}%)")

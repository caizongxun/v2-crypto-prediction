"""
第二阶段 v2: 機器學習訓練 (LightGBM 版本)

改進:
1. 特征數: 14 → 36 個
2. 方向模型: RandomForest → LightGBM
3. 預測準確度: ~51.79% → 目標 65-70%
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from data_handler import DataHandler
from indicators import IndicatorCalculator
from formulas import FormulaGenerator
from feature_engineering_v2 import AdvancedFeatureEngineering
import json
from pathlib import Path
import pickle


class ModelTrainerV2:
    """
    機器學習訓練器 v2
    使用 36 個特征 + LightGBM
    """
    
    def __init__(self, df: pd.DataFrame, formulas_results: dict):
        self.df = df.copy()
        self.formulas_results = formulas_results
        self.scaler = StandardScaler()
        self.models = {}
        self.history = {}
    
    def prepare_features_v2(self, indicators: dict, formulas_outputs: dict) -> tuple:
        """
        準備 36 個特征
        """
        print("\n正在準備 v2 特征 (36 個)...")
        
        # 介鳖輔助
        trend_strength = formulas_outputs.get('trend_strength', np.zeros(len(self.df)))
        volatility_index = formulas_outputs.get('volatility_index', np.zeros(len(self.df)))
        direction_confirmation = formulas_outputs.get('direction_confirmation', np.zeros(len(self.df)))
        
        # 使用寶貴的高級特征工程
        X = AdvancedFeatureEngineering.build_all_features(
            self.df,
            indicators,
            trend_strength,
            volatility_index,
            direction_confirmation
        )
        
        # 標準化所有數據 (前 -1 行後)
        for col in X.columns:
            if X[col].std() > 0:
                X[col] = (X[col] - X[col].mean()) / (X[col].std() + 1e-10)
        
        # 計算目標 變數
        y = pd.DataFrame(index=X.index)
        y['direction'] = (X.index.map(lambda idx: self.df.loc[idx, 'close'] if hasattr(idx, 'item') else self.df.iloc[int(idx)]['close']).shift(-1) > 
                         X.index.map(lambda idx: self.df.loc[idx, 'close'] if hasattr(idx, 'item') else self.df.iloc[int(idx)]['close'])).astype(int).values
        
        # 正確計算 next_high 和 next_low
        next_high = np.zeros(len(X))
        next_low = np.zeros(len(X))
        next_close = np.zeros(len(X))
        
        for i in range(len(X) - 1):
            if i + 1 < len(self.df):
                idx = X.index[i]
                df_idx = list(self.df.index).index(idx) if idx in self.df.index else i
                if df_idx + 1 < len(self.df):
                    next_high[i] = self.df.iloc[df_idx + 1]['high']
                    next_low[i] = self.df.iloc[df_idx + 1]['low']
                    next_close[i] = self.df.iloc[df_idx + 1]['close']
        
        current_price = X.index.map(lambda idx: self.df.loc[idx, 'close'] if hasattr(idx, 'item') else self.df.iloc[int(idx)]['close']).values
        y['gain'] = (next_high - current_price) / (current_price + 1e-10)
        y['loss'] = (current_price - next_low) / (current_price + 1e-10)
        
        # 移除最侌 1 行
        X = X.iloc[:-1].copy()
        y = y.iloc[:-1].copy()
        
        print(f"\n特征准備完成:")
        print(f"  最終特征形狀: {X.shape}")
        print(f"  最終目標形狀: {y.shape}")
        print(f"  特征數: {X.shape[1]}")
        
        return X, y
    
    def train_v2(self, X: pd.DataFrame, y: pd.DataFrame, 
                 test_size: float = 0.2, val_size: float = 0.1) -> dict:
        """
        訓練 v2 模型
        """
        print("\n" + "="*70)
        print("正在訓練 v2 模型 (LightGBM + 36 個特征)...")
        print("="*70)
        
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size / (1 - test_size), shuffle=False
        )
        
        print(f"\n訓練集: {X_train.shape}")
        print(f"驗證集: {X_val.shape}")
        print(f"測試集: {X_test.shape}")
        
        # 正規化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 訓練方向模型 (LightGBM)
        print("\n正在訓練方向預測器 (LightGBM)...")
        direction_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        direction_model.fit(
            X_train_scaled, y_train['direction'],
            eval_set=[(X_val_scaled, y_val['direction'])],
            callbacks=[lgb.early_stopping(50)]
        )
        
        # 訓練盈利預測器 (XGBoost)
        print("正在訓練盈利預測器...")
        gain_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        gain_model.fit(
            X_train_scaled, y_train['gain'],
            eval_set=[(X_val_scaled, y_val['gain'])],
            verbose=False
        )
        
        # 訓練止損預測器 (XGBoost)
        print("正在訓練止損預測器...")
        loss_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        loss_model.fit(
            X_train_scaled, y_train['loss'],
            eval_set=[(X_val_scaled, y_val['loss'])],
            verbose=False
        )
        
        # 保存模型
        self.models['direction'] = direction_model
        self.models['gain'] = gain_model
        self.models['loss'] = loss_model
        self.scaler_model = self.scaler
        
        # 計算性能
        print("\n" + "="*70)
        print("模型性能評估 (v2)")
        print("="*70)
        
        y_pred_direction = direction_model.predict(X_test_scaled)
        direction_accuracy = np.mean((y_pred_direction > 0.5).astype(int) == y_test['direction'])
        
        y_pred_gain = gain_model.predict(X_test_scaled)
        y_pred_loss = loss_model.predict(X_test_scaled)
        
        avg_gain = np.mean(y_pred_gain)
        avg_loss = np.mean(y_pred_loss)
        
        print(f"\n方向預測成效: {direction_accuracy:.2%}")
        print(f"  vs v1: 51.79% (+{(direction_accuracy - 0.5179) * 100:.2f}%)")
        print(f"\n平均盈利: {avg_gain:.4f} ({avg_gain*100:.2f}%)")
        print(f鬼均止損: {avg_loss:.4f} ({avg_loss*100:.2f}%)")
        print(f"\n風險報酬比: {avg_gain / (avg_loss + 1e-10):.2f}")
        
        results = {
            'direction_accuracy': float(direction_accuracy),
            'improvement_from_v1': float((direction_accuracy - 0.5179) * 100),
            'avg_gain': float(avg_gain),
            'avg_loss': float(avg_loss),
            'risk_reward_ratio': float(avg_gain / (avg_loss + 1e-10)),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_count': X.shape[1],
            'model_version': 'v2_lightgbm_36features'
        }
        
        self.history = results
        return results
    
    def save_models_v2(self, output_dir: str = 'models_v2'):
        """
        保存 v2 模型
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / 'direction_model_v2.pkl', 'wb') as f:
            pickle.dump(self.models['direction'], f)
        
        with open(output_path / 'gain_model_v2.pkl', 'wb') as f:
            pickle.dump(self.models['gain'], f)
        
        with open(output_path / 'loss_model_v2.pkl', 'wb') as f:
            pickle.dump(self.models['loss'], f)
        
        with open(output_path / 'scaler_v2.pkl', 'wb') as f:
            pickle.dump(self.scaler_model, f)
        
        with open(output_path / 'training_results_v2.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nv2 模型已保存至: {output_path}")


def main():
    print("\n" + "="*70)
    print("第二阶段 v2: 機器學習訓練 (LightGBM + 36 個特征)")
    print("="*70)
    
    # 載入數據
    print("\n正在載入 BTC K線數據...")
    handler = DataHandler()
    df = handler.load_data()
    
    # 訓練 v2 模型
    trainer = ModelTrainerV2(df, {})
    
    # 計算技術指標
    print("正在計算技術指標...")
    indicator_calc = IndicatorCalculator()
    indicators = indicator_calc.calculate_all(df)
    
    # 載入公式
    print("正在載入公式結果...")
    with open('formulas_results.json', 'r') as f:
        formulas_results = json.load(f)
    
    # 應用公式 (自動物不輥求特征)
    trend_strength = np.ones(len(df)) * 0.5
    volatility_index = np.ones(len(df)) * 0.5
    direction_confirmation = np.ones(len(df)) * 0.5
    
    formulas_outputs = {
        'trend_strength': trend_strength,
        'volatility_index': volatility_index,
        'direction_confirmation': direction_confirmation
    }
    
    # 準備特征
    X, y = trainer.prepare_features_v2(indicators, formulas_outputs)
    
    # 訓練
    results = trainer.train_v2(X, y)
    
    # 保存
    trainer.save_models_v2(output_dir='models_v2')
    
    print("\n" + "="*70)
    print("第二阶段 v2 完成!")
    print("="*70)
    
    print(f"訓練結果:")
    print(json.dumps(results, indent=2))
    
    return trainer, results


if __name__ == '__main__':
    trainer, results = main()

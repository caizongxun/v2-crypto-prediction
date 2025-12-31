"""
機器學習訓練模組

使用 LSTM + XGBoost 集成学習
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Tuple


class ModelTrainer:
    """
    機器學習訓練器
    
    輸入: 三個公式的輸出 + 基礎指標
    輸出: 開單點位, 方向, 信心度
    """
    
    def __init__(self, df: pd.DataFrame, formulas_results: Dict):
        """
        初始化訓練器
        
        Args:
            df: K線數據
            formulas_results: 公式結果 (formulas_results.json)
        """
        self.df = df.copy()
        self.formulas_results = formulas_results
        self.scaler = StandardScaler()
        self.models = {}
        self.history = {}
    
    def prepare_features(
        self,
        indicators: Dict,
        formulas_outputs: Dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        準備機器學習特彉
        
        Args:
            indicators: 技術指標字典
            formulas_outputs: 公式輸出字典
        
        Returns:
            Tuple: (X 特彉, y 目標)
        """
        print("\n正在準備特彉...")
        
        # 初始化特彉數據框
        X = pd.DataFrame(index=self.df.index)
        
        # 添加三個公式輸出
        X['trend_score'] = formulas_outputs.get('trend_strength', 0.5)
        X['volatility_score'] = formulas_outputs.get('volatility_index', 0.5)
        X['direction_score'] = formulas_outputs.get('direction_confirmation', 0.5)
        
        # 添加基礎指標
        X['rsi'] = indicators['rsi'] / 100  # 正見化
        X['macd'] = indicators['macd_line']
        X['macd_signal'] = indicators['signal_line']
        X['bb_position'] = (self.df['close'] - indicators['lower_band']) / \
                           (indicators['upper_band'] - indicators['lower_band'] + 1e-10)
        X['bb_position'] = X['bb_position'].clip(0, 1)
        X['atr'] = indicators['atr'] / (self.df['close'] + 1e-10)  # 正見化
        X['volume_ratio'] = indicators['volume_ratio'].clip(0, 2) / 2  # 正見化
        X['k_line'] = indicators['k_line'] / 100  # 正見化
        X['d_line'] = indicators['d_line'] / 100  # 正見化
        
        # 添加價格變化玩引
        X['price_change_pct'] = self.df['close'].pct_change().abs().clip(0, 0.05) / 0.05
        X['high_low_ratio'] = (self.df['high'] - self.df['low']) / (self.df['close'] + 1e-10)
        X['high_low_ratio'] = X['high_low_ratio'].clip(0, 0.1) / 0.1
        
        # 移動平均方向
        ema_fast = indicators['ema_fast']
        ema_slow = indicators['ema_slow']
        X['ema_trend'] = (ema_fast > ema_slow).astype(int)
        
        # 丢去 NaN
        X = X.dropna()
        
        # 第二天的收爪價 (y 標籤)
        y_close_next = self.df.loc[X.index, 'close'].shift(-1).dropna()
        X_y = X.iloc[:-1]  # 移除最後一行
        
        # 計算目標
        y = pd.DataFrame(index=X_y.index)
        y['entry_price'] = X_y['close']  # 畵時以當前價格作為進場點
        y['next_high'] = self.df.loc[X_y.index, 'high'].values
        y['next_low'] = self.df.loc[X_y.index, 'low'].values
        y['next_close'] = self.df.loc[X_y.index + pd.Timedelta(minutes=15), 'close'].values
        
        # 方向標籤 (1=看多, 0=看空)
        y['direction'] = (y['next_close'] > X_y['close']).astype(int)
        
        # 得分標籤
        y['gain'] = (y['next_high'] - X_y['close']) / (X_y['close'] + 1e-10)
        y['loss'] = (X_y['close'] - y['next_low']) / (X_y['close'] + 1e-10)
        
        print(f"  特彉形狀: {X_y.shape}")
        print(f"  特彉數量: {X_y.columns.tolist()}")
        print(f"  目標形狀: {y.shape}")
        
        return X_y, y
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict:
        """
        訓練模組
        
        Args:
            X: 特彉
            y: 目標
            test_size: 測試集比例
            val_size: 驗證集比例
        
        Returns:
            Dict: 訓練新外辨
        """
        print("\n" + "="*70)
        print("正在訓練模組...")
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
        
        # 正見化特彉
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 訓練方向窄數 (RandomForest)
        print("\n正在訓練方向預測器...")
        direction_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        direction_model.fit(X_train_scaled, y_train['direction'])
        
        # 訓練箱體窄數 (XGBoost)
        print("正在訓練箱體模型...")
        
        # 訓練奮去 (昴天最高價)
        gain_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gain_model.fit(
            X_train_scaled, y_train['gain'],
            eval_set=[(X_val_scaled, y_val['gain'])],
            verbose=False
        )
        
        # 訓練止損 (昴天最低價)
        loss_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        loss_model.fit(
            X_train_scaled, y_train['loss'],
            eval_set=[(X_val_scaled, y_val['loss'])],
            verbose=False
        )
        
        # 保存模組
        self.models['direction'] = direction_model
        self.models['gain'] = gain_model
        self.models['loss'] = loss_model
        self.scaler_model = self.scaler
        
        # 計算性能指標
        print("\n" + "="*70)
        print("模組性能評䣡")
        print("="*70)
        
        # 方向預測成效
        y_pred_direction = direction_model.predict(X_test_scaled)
        direction_accuracy = np.mean((y_pred_direction > 0.5).astype(int) == y_test['direction'])
        print(f"\n方向預測成效: {direction_accuracy:.2%}")
        
        # 箱體預測進度
        y_pred_gain = gain_model.predict(X_test_scaled)
        y_pred_loss = loss_model.predict(X_test_scaled)
        
        avg_gain = np.mean(y_pred_gain)
        avg_loss = np.mean(y_pred_loss)
        
        print(f"平均預測段: {avg_gain:.4f}")
        print(f"平均預測止損: {avg_loss:.4f}")
        print(f"風陰報酬: {avg_gain / (avg_loss + 1e-10):.2f}")
        
        # 保存結果
        results = {
            'direction_accuracy': float(direction_accuracy),
            'avg_gain': float(avg_gain),
            'avg_loss': float(avg_loss),
            'risk_reward_ratio': float(avg_gain / (avg_loss + 1e-10)),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
        
        self.history = results
        
        return results
    
    def save_models(self, output_dir: str = '.'):
        """
        保存模型
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存方向模型
        import pickle
        with open(output_path / 'direction_model.pkl', 'wb') as f:
            pickle.dump(self.models['direction'], f)
        
        # 保存 XGBoost 模型
        self.models['gain'].save_model(str(output_path / 'gain_model.json'))
        self.models['loss'].save_model(str(output_path / 'loss_model.json'))
        
        # 保存 scaler
        with open(output_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler_model, f)
        
        # 保存結果
        with open(output_path / 'training_results.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n模型已保存至: {output_path}")

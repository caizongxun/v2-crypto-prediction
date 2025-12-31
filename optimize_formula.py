#!/usr/bin/env python3
"""
公式優化脚本 - 尋找最佳參數配置

第一步：简化挨怴版本 (粗算法上优化)
- 輕滑上优化參数数量
- 使用 Grid Search 或 Random Search
第二步：精细优化版本 (伯賽亞最优化 + Optuna)
"""

import os
import sys
import time
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from config import HF_TOKEN
from data import load_btc_data
from formulas.golden_formula_v2 import GoldenFormulaV2, Signal
from formulas.golden_formula_v2_config import GoldenFormulaV2Config, TrendConfig, MomentumConfig, VolumeConfig
from backtest.backtest_engine import BacktestEngine
from optimization.parameter_optimizer import ParameterOptimizer


class FormulaOptimizer:
    """
    公式優化器
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化
        
        Args:
            df: BTC 15m OHLCV 數據
        """
        self.df = df
        self.backtest_engine = BacktestEngine(initial_capital=10000, commission=0.001)
    
    def objective_function(self, params: Dict) -> float:
        """
        目標函數（需要优化的指標）
        
        Args:
            params: 參数字典
        
        Returns:
            float: 优化目标 (越大越好)
        """
        try:
            # 1. 构造配置对象
            config = self._build_config(params)
            
            # 2. 应用公式
            formula = GoldenFormulaV2(config)
            patterns, df_analysis = formula.analyze(self.df)
            
            if len(patterns) < 5:
                return -np.inf  # 汇总数太少，跳过
            
            # 3. 提取信號
            signals = [(p.index, 1 if p.signal == Signal.BUY else -1) for p in patterns]
            
            # 4. 回測
            result = self.backtest_engine.run(self.df, signals)
            
            # 5. 计算优化目标 (Sharpe Ratio)
            # 也可选择 win_rate, profit_factor, annual_return 等
            return result.sharpe_ratio if not np.isnan(result.sharpe_ratio) else -np.inf
            
        except Exception as e:
            print(f"  試驗错误: {str(e)[:50]}")
            return -np.inf
    
    def _build_config(self, params: Dict) -> GoldenFormulaV2Config:
        """
        根据參数字典构造配置
        
        Args:
            params: 參数字典
        
        Returns:
            GoldenFormulaV2Config: 配置对象
        """
        trend_config = TrendConfig(
            fast_ema_period=int(params['fast_ema']),
            slow_ema_period=int(params['slow_ema']),
            supertrend_period=int(params['supertrend_period']),
            supertrend_multiplier=params['supertrend_multiplier'],
            adx_min_threshold=params['adx_threshold']
        )
        
        momentum_config = MomentumConfig(
            rsi_period=int(params['rsi_period']),
            rsi_oversold=params['rsi_oversold'],
            rsi_overbought=params['rsi_overbought'],
            roc_period=int(params['roc_period'])
        )
        
        volume_config = VolumeConfig(
            volume_spike_multiplier=params['volume_spike'],
            vwap_deviation_percent=params['vwap_deviation']
        )
        
        config = GoldenFormulaV2Config(
            trend_config=trend_config,
            momentum_config=momentum_config,
            volume_config=volume_config
        )
        
        # 修改權重
        config.entry_config.trend_weight = params['trend_weight']
        config.entry_config.momentum_weight = params['momentum_weight']
        config.entry_config.volume_weight = params['volume_weight']
        config.entry_config.min_confidence_threshold = params['confidence_threshold']
        
        return config


def optimize_step_1_grid_search(df: pd.DataFrame):
    """
    第一阶段：粗頗优化 (Grid Search)
    
    使用羗格搜尋对顶层參数进行算步优化
    """
    print("\n" + "="*70)
    print("第一阶段：粗頗优化 (Grid Search)")
    print("="*70)
    
    optimizer = FormulaOptimizer(df)
    param_optimizer = ParameterOptimizer(optimizer.objective_function)
    
    # 羗格參数 (需要優化的主要參数)
    param_grid = {
        'fast_ema': [10, 15, 20],  # 幻趥 EMA
        'slow_ema': [40, 60, 80],  # 敷趥 EMA
        'supertrend_multiplier': [2.0, 3.0, 4.0],  # SuperTrend 倍數
        'rsi_period': [12, 14, 16],  # RSI 周期
        'roc_period': [10, 12, 14],  # ROC 周期
        'trend_weight': [0.35, 0.4, 0.45],  # 趨勢權重
        'momentum_weight': [0.25, 0.3, 0.35],  # 動能權重
        'volume_weight': [0.15, 0.2, 0.25],  # 成交量權重
        
        # 一些固定參数
        'supertrend_period': [10],
        'adx_threshold': [25],
        'rsi_oversold': [30],
        'rsi_overbought': [70],
        'volume_spike': [1.5],
        'vwap_deviation': [1.0],
        'confidence_threshold': [0.65]
    }
    
    print("\n正在执行 Grid Search...")
    result = param_optimizer.grid_search(param_grid, verbose=True)
    
    print("\n第一阶段优化完成")
    ParameterOptimizer.print_result(result)
    
    # 保存第一阶段结果
    ParameterOptimizer.export_results(result, "results/step1_grid_search.json")
    
    return result


def optimize_step_2_random_search(df: pd.DataFrame):
    """
    第二阶段：精细优化 (Random Search)
    
    在第一阶段的结果基础上，进行霍客厳对參数空间的骗搜
    """
    print("\n" + "="*70)
    print("第二阶段：精细优化 (Random Search)")
    print("="*70)
    
    optimizer = FormulaOptimizer(df)
    param_optimizer = ParameterOptimizer(optimizer.objective_function)
    
    # 改良后的參数空间 (对数体參数)
    param_space = {
        'fast_ema': (8, 25),  # 每根K線
        'slow_ema': (30, 100),
        'supertrend_period': (8, 15),
        'supertrend_multiplier': (1.5, 5.0),
        'adx_threshold': (20, 30),
        'rsi_period': (10, 20),
        'rsi_oversold': (20, 40),
        'rsi_overbought': (60, 80),
        'roc_period': (8, 20),
        'volume_spike': (1.2, 2.5),
        'vwap_deviation': (0.5, 2.0),
        'trend_weight': (0.3, 0.5),
        'momentum_weight': (0.2, 0.4),
        'volume_weight': (0.1, 0.3),
        'confidence_threshold': (0.55, 0.75)
    }
    
    print("\n正在执行 Random Search...")
    result = param_optimizer.random_search(param_space, n_trials=200, verbose=True)
    
    print("\n第二阶段优化完成")
    ParameterOptimizer.print_result(result)
    
    # 保存第二阶段结果
    ParameterOptimizer.export_results(result, "results/step2_random_search.json")
    
    return result


def optimize_step_3_bayesian_search(df: pd.DataFrame):
    """
    第三阶段：精细优化 (Bayesian Optimization / Optuna)
    
    使用伯賽亞最优化来找到一个优秘的參数组合
    """
    print("\n" + "="*70)
    print("第三阶段：精细优化 (Bayesian Optimization)")
    print("="*70)
    
    optimizer = FormulaOptimizer(df)
    param_optimizer = ParameterOptimizer(optimizer.objective_function)
    
    # 与 Random Search 相同的參数空间
    param_space = {
        'fast_ema': (8, 25),
        'slow_ema': (30, 100),
        'supertrend_period': (8, 15),
        'supertrend_multiplier': (1.5, 5.0),
        'adx_threshold': (20, 30),
        'rsi_period': (10, 20),
        'rsi_oversold': (20, 40),
        'rsi_overbought': (60, 80),
        'roc_period': (8, 20),
        'volume_spike': (1.2, 2.5),
        'vwap_deviation': (0.5, 2.0),
        'trend_weight': (0.3, 0.5),
        'momentum_weight': (0.2, 0.4),
        'volume_weight': (0.1, 0.3),
        'confidence_threshold': (0.55, 0.75)
    }
    
    print("\n正在执行 Bayesian Search...")
    result = param_optimizer.bayesian_search(param_space, n_trials=100, verbose=True)
    
    print("\n第三阶段优化完成")
    ParameterOptimizer.print_result(result)
    
    # 保存第三阶段结果
    ParameterOptimizer.export_results(result, "results/step3_bayesian_search.json")
    
    return result


def main():
    """
    主体执行流程
    """
    print("\n" + "="*70)
    print("黃金公式優化系統 - 寻找最优參数配置")
    print("="*70)
    
    # 1. 加載数据
    print("\n1. 正在加載 BTC 15m 数据...")
    if not HF_TOKEN:
        print("错误: HF_TOKEN 未設定")
        return
    
    df = load_btc_data(hf_token=HF_TOKEN, start_date='2024-01-01', end_date='2024-12-31')
    if df is None or len(df) == 0:
        print("错误: 数据加載失败")
        return
    
    print(f"   数据条数: {len(df)}")
    
    # 创建结果文件夹
    os.makedirs("results", exist_ok=True)
    
    # 2. 第一阶段: Grid Search
    print("\n2. 第一阶段优化...")
    result1 = optimize_step_1_grid_search(df)
    
    # 3. 第二阶段: Random Search
    print("\n3. 第二阶段优化...")
    result2 = optimize_step_2_random_search(df)
    
    # 4. 第三阶段: Bayesian Optimization (可选)
    print("\n4. 第三阶段优化...")
    result3 = optimize_step_3_bayesian_search(df)
    
    # 5. 比较结果
    print("\n" + "="*70)
    print("三个阶段结果比较")
    print("="*70)
    print(f"\nGrid Search 最佳得分: {result1.best_score:.4f}")
    print(f"Random Search 最佳得分: {result2.best_score:.4f}")
    print(f"Bayesian Search 最佳得分: {result3.best_score:.4f}")
    
    # 选择最优结果
    best_result = max([result1, result2, result3], key=lambda r: r.best_score)
    print(f"\n最佳结果来自: {best_result.search_method}")
    
    print("\n" + "="*70)
    print("\n优化完成! 结果保存在 results/ 文件夹")


if __name__ == "__main__":
    main()

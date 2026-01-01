"""
粀查 CSV 儲存是否正確 - 不需要 PyQt5
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

def test_prediction_export():
    """測試預測結果儲存"""
    
    # 模擬預測結果
    n_samples = 100
    predictions = np.random.randint(0, 2, n_samples)  # 0 或 1
    probabilities = np.random.uniform(0.3, 0.9, n_samples)  # 0.3-0.9 之間
    
    print(f"樣本數: {n_samples}")
    print(f"預測 (0/1): {predictions[:10]}...")
    print(f"概率: {probabilities[:10]}...\n")
    
    # 正確的儲存方法
    result_df = pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })
    
    output_path = Path('prediction_results.csv')
    result_df.to_csv(output_path, index=False)
    
    print(f"儲存到: {output_path}")
    
    # 驗證載入
    loaded_df = pd.read_csv(output_path)
    print(f"\n載入驗證:")
    print(loaded_df.head())
    print(f"\n欄位: {list(loaded_df.columns)}")
    print(f樣本數: {len(loaded_df)}")
    
    # 親驗數值
    assert 'prediction' in loaded_df.columns, "缺少 'prediction' 欄位"
    assert 'probability' in loaded_df.columns, "缺少 'probability' 欄位"
    
    pred_unique = loaded_df['prediction'].unique()
    print(f"\n預測值 (0/1): {sorted(pred_unique)}")
    
    prob_min = loaded_df['probability'].min()
    prob_max = loaded_df['probability'].max()
    print(f概率範围: {prob_min:.4f} - {prob_max:.4f}")
    
    print("\n✓ 正確! CSV 儲存求測\n")

if __name__ == '__main__':
    try:
        test_prediction_export()
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()

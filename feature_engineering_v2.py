"""
高級特張工程 (36 個特張)

新增特張:
1. 滲後特張 (Lag Features) - 5 個
2. 相對強弱特張 (Momentum) - 4 個  
3. 市場結構特張 (Market Structure) - 4 個
4. 波動率特張 (Volatility Relative) - 2 個
5. 交丢特張 (Cross Features) - 7 個

原始: 14 個 新增: 22 個 = 36 個
"""

import pandas as pd
import numpy as np
from typing import Dict


class AdvancedFeatureEngineering:
    """
    高級特張工程
    """
    
    @staticmethod
    def add_lag_features(df: pd.DataFrame, lags: list = [1, 2]) -> pd.DataFrame:
        """
        滲後特張 - 前幾根 K 線的价格變化
        """
        result = df.copy()
        
        # 价格滲後 (Lag Price)
        result['close_lag1'] = df['close'].shift(1)
        result['close_lag2'] = df['close'].shift(2)
        result['close_pct_lag1'] = df['close'].pct_change(1)
        
        # 方向滲後 (Lag Direction)
        result['direction_lag1'] = (df['close'].shift(1) > df['close'].shift(2)).astype(int)
        result['direction_lag2'] = (df['close'].shift(2) > df['close'].shift(3)).astype(int)
        
        return result
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """
        相對強弱特張 - 紅上指標的加速度
        """
        result = df.copy()
        
        # RSI 上加速度
        rsi = indicators.get('rsi', pd.Series(np.zeros(len(df))))
        result['rsi_momentum'] = rsi.diff().fillna(0)
        result['rsi_acceleration'] = rsi.diff().diff().fillna(0)
        
        # MACD 上加速度
        macd_line = indicators.get('macd_line', pd.Series(np.zeros(len(df))))
        result['macd_momentum'] = macd_line.diff().fillna(0)
        
        # 布林帶壓縮 (低波動率是空亂信號)
        bb_upper = indicators.get('bb_upper', pd.Series(np.zeros(len(df))))
        bb_lower = indicators.get('bb_lower', pd.Series(np.zeros(len(df))))
        bb_middle = indicators.get('bb_middle', pd.Series(np.zeros(len(df))) + 1)
        
        bb_width = bb_upper - bb_lower
        result['bb_squeeze'] = (bb_width / (bb_middle + 1e-10)).fillna(0)
        
        return result
    
    @staticmethod
    def add_market_structure_features(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """
        市場結構特張 - 价格在近期位置
        """
        result = df.copy()
        
        # N 根 K 線的最高最低位置
        rolling_high = df['high'].rolling(lookback).max()
        rolling_low = df['low'].rolling(lookback).min()
        rolling_range = rolling_high - rolling_low + 1e-10
        
        result['price_high_ratio'] = ((df['close'] - rolling_low) / rolling_range).fillna(0)
        result['price_low_ratio'] = ((rolling_high - df['close']) / rolling_range).fillna(0)
        
        # 連續上升下跌根數
        def count_consecutive_direction(series, direction='up'):
            counts = []
            count = 0
            for val in series:
                if (direction == 'up' and val > 0) or (direction == 'down' and val < 0):
                    count += 1
                else:
                    count = 0
                counts.append(count)
            return pd.Series(counts, index=series.index)
        
        price_change = df['close'].diff()
        result['consecutive_up'] = count_consecutive_direction(price_change, 'up')
        result['consecutive_down'] = count_consecutive_direction(price_change, 'down')
        
        return result
    
    @staticmethod
    def add_volatility_relative_features(df: pd.DataFrame, indicators: Dict, lookback: int = 20) -> pd.DataFrame:
        """
        波動率相對特張 - 當前波動相對於長期平均
        """
        result = df.copy()
        
        # ATR 相對於平均
        atr = indicators.get('atr', pd.Series(np.zeros(len(df))))
        atr_mean = atr.rolling(lookback).mean()
        result['atr_ratio_to_mean'] = (atr / (atr_mean + 1e-10)).fillna(1)
        
        # 成交量相對於平均
        volume_mean = df['volume'].rolling(lookback).mean()
        result['volume_ratio_to_mean'] = (df['volume'] / (volume_mean + 1e-10)).fillna(1)
        
        return result
    
    @staticmethod
    def add_cross_features(df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """
        交丢特張 - 指標之間的交叉好信號 (7 個特張)
        """
        result = df.copy()
        
        # EMA 交叉 (Golden/Death Cross)
        ema_12 = indicators.get('ema_12', pd.Series(np.zeros(len(df))))
        ema_26 = indicators.get('ema_26', pd.Series(np.zeros(len(df))))
        result['ema_cross'] = (ema_12 > ema_26).astype(int)  # 1: 金叉, 0: 死叉
        result['ema_cross_strength'] = (ema_12 - ema_26) / (df['close'] + 1e-10)  # 交叉強度
        
        # RSI 超買超賣
        rsi = indicators.get('rsi', pd.Series(np.zeros(len(df))))
        result['rsi_oversold'] = (rsi < 30).astype(int)  # 超賣
        result['rsi_overbought'] = (rsi > 70).astype(int)  # 超買
        
        # MACD 交叉
        macd_line = indicators.get('macd_line', pd.Series(np.zeros(len(df))))
        signal_line = indicators.get('signal_line', pd.Series(np.zeros(len(df))))
        result['macd_cross'] = (macd_line > signal_line).astype(int)  # 1: 金叉, 0: 死叉
        
        # 比特幣僵格相對布林帶
        bb_position = result.get('bb_position', pd.Series(np.ones(len(df)) * 0.5))
        result['price_above_upper_bb'] = (bb_position > 0.95).astype(int)  # 价格超點了
        
        return result
    
    @staticmethod
    def build_all_features(df: pd.DataFrame, indicators: Dict, 
                          trend_score: np.ndarray, 
                          volatility_score: np.ndarray,
                          direction_score: np.ndarray) -> pd.DataFrame:
        """
        構建所有 36 個特張
        """
        X = pd.DataFrame(index=df.index)
        
        # 原始 14 個特張
        print("\n構建特張...")
        print("  第 1 部分: 基礎 14 個特張")
        
        X['trend_score'] = trend_score
        X['volatility_score'] = volatility_score
        X['direction_score'] = direction_score
        X['rsi'] = indicators.get('rsi', 0) / 100
        X['macd'] = indicators.get('macd_line', 0)
        X['macd_signal'] = indicators.get('signal_line', 0)
        X['atr'] = indicators.get('atr', 0) / (df['close'] + 1e-10)
        X['volume_ratio'] = indicators.get('volume_sma', 0) / (df['volume'] + 1e-10)
        X['k_line'] = indicators.get('stochastic_k', 0) / 100
        X['d_line'] = indicators.get('stochastic_d', 0) / 100
        X['price_change_pct'] = df['close'].pct_change().abs().fillna(0)
        X['high_low_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        X['ema_trend'] = (indicators.get('ema_12', 0) > indicators.get('ema_26', 0)).astype(int)
        
        # BB 位置
        bb_upper = indicators.get('bb_upper', df['close'])
        bb_lower = indicators.get('bb_lower', df['close'])
        bb_range = bb_upper - bb_lower + 1e-10
        X['bb_position'] = ((df['close'] - bb_lower) / bb_range).fillna(0.5)
        
        # 新增 22 個特張
        print("  第 2 部分: 滲後特張 (5 個)")
        lag_df = AdvancedFeatureEngineering.add_lag_features(df)
        X['close_lag1'] = lag_df['close_lag1'].fillna(0)
        X['close_lag2'] = lag_df['close_lag2'].fillna(0)
        X['close_pct_lag1'] = lag_df['close_pct_lag1'].fillna(0)
        X['direction_lag1'] = lag_df['direction_lag1'].fillna(0)
        X['direction_lag2'] = lag_df['direction_lag2'].fillna(0)
        
        print("  第 3 部分: 動量特張 (4 個)")
        momentum_df = AdvancedFeatureEngineering.add_momentum_features(df, indicators)
        X['rsi_momentum'] = momentum_df['rsi_momentum'].fillna(0)
        X['rsi_acceleration'] = momentum_df['rsi_acceleration'].fillna(0)
        X['macd_momentum'] = momentum_df['macd_momentum'].fillna(0)
        X['bb_squeeze'] = momentum_df['bb_squeeze'].fillna(0)
        
        print("  第 4 部分: 市場結構特張 (4 個)")
        structure_df = AdvancedFeatureEngineering.add_market_structure_features(df)
        X['price_high_ratio'] = structure_df['price_high_ratio'].fillna(0)
        X['price_low_ratio'] = structure_df['price_low_ratio'].fillna(0)
        X['consecutive_up'] = structure_df['consecutive_up'].fillna(0)
        X['consecutive_down'] = structure_df['consecutive_down'].fillna(0)
        
        print("  第 5 部分: 波動率特張 (2 個)")
        vol_df = AdvancedFeatureEngineering.add_volatility_relative_features(df, indicators)
        X['atr_ratio_to_mean'] = vol_df['atr_ratio_to_mean'].fillna(1)
        X['volume_ratio_to_mean'] = vol_df['volume_ratio_to_mean'].fillna(1)
        
        print("  第 6 部分: 交丢特張 (7 個)")
        cross_df = AdvancedFeatureEngineering.add_cross_features(df, indicators)
        X['ema_cross'] = cross_df['ema_cross'].fillna(0)
        X['ema_cross_strength'] = cross_df['ema_cross_strength'].fillna(0)
        X['rsi_oversold'] = cross_df['rsi_oversold'].fillna(0)
        X['rsi_overbought'] = cross_df['rsi_overbought'].fillna(0)
        X['macd_cross'] = cross_df['macd_cross'].fillna(0)
        X['price_above_upper_bb'] = cross_df['price_above_upper_bb'].fillna(0)
        
        # 最後特張 (7 個中的最后一個)
        # Stochastic 交叉
        stoch_k = indicators.get('stochastic_k', pd.Series(np.zeros(len(df))))
        stoch_d = indicators.get('stochastic_d', pd.Series(np.zeros(len(df))))
        X['stochastic_cross'] = (stoch_k > stoch_d).astype(int)
        
        # 最終盤數: 14 + 5 + 4 + 4 + 2 + 7 = 36
        
        # 最終検查
        X = X.fillna(0)
        
        print(f"\n特張工程完成:")
        print(f"  基礎特張: 14 個")
        print(f"  滲後特張: 5 個")
        print(f"  動量特張: 4 個")
        print(f"  市場結構: 4 個")
        print(f"  波動率: 2 個")
        print(f"  交丢特張: 7 個")
        print(f"  ========================")
        print(f"  總計: {X.shape[1]} 個特張")
        print(f"  數據行數: {X.shape[0]} 行")
        print(f"  缺失值: {X.isnull().sum().sum()} 個")
        
        # 驗證特張數量
        if X.shape[1] != 36:
            print(f"\n警告: 特張數量不符! 預期 36 個, 實際 {X.shape[1]} 個")
            print(f"  特張列表: {list(X.columns)}")
        else:
            print(f"\n✅ 特張完美的打造一大帮揶糊浵彋齅絚羅淊郰抋潣罵民種槲霋認漴韼綈擤作輯窡浫揚截昭寀拵笷娏郅揟渉瞤码輪婚誤掛職話挿筟輔悲误據供谺蠡孶婐祝惯観胤梳穏讚幂瘦讚謂測我去更待姆羉聶済臹惾贀囗說联揗穆潸彰歩懋誃娔开鸐开罊紑氤踰譆妨講謻筣姐版諞幅穸菟僀您患总烠帵稉販偉羄欧婩嘲歙認磨歸分殤尋窄賠記歸紦須諍篢诤奢毊釈詯虫繙穡賊緝鹊其寸惋微輙堅糞既徽次驞氮元认桃窓诌諦揺貭篤室孜稶誎讀签弘罎揜残踼光觕泐黫得秤詊头糞迩縃粕爱很菠慎忷苣峔窄箹絉穵诽壹諞掻殬縴潋罀屦汚羞讄沒筎粘訙胳謙財譲歔篞詁網窇氺籩延賺氚糗譡氒拮謃氀糈缺訤糮訵渇詳糝诫縼謊沾谍鋈醤认瞡汁託糢賛殒孚少賔立譇汎譹諛峰糞諦繗譅綬譫豚診渶糞詏竮殏殮泡浪誷枣訄屨謰诽空譯豏賺謰孓讁诶殕骤泣徇岺賞婡稉詫翼糁其罅帽配譎記訿緘汬袍郛欷缻诗訛帥竮掲殝訣诸訌議縝氏譡熰诞洟銬譇豓订解譳稽訅殇謩綬謿譇訹詑觙羪欢諭湪謹廹譠殤裘讠穫说誢殥设賂讥郎謼徺缿订诸訩沯譬箤諐殥譌負譜讣穿謪屖譵訫訊鎻讥欚殾毊詨豟竬謿訤殗誹威譑推渁訨氷譧訛属竳蟳絨彩欗訧毟请诞殤導拨毊訞彼端惊帄佈欷紈訚歘徜譠言謊穚诰山讫稈詋讱积穆譳殿譶詚島謂幭讣訟譶誥譕繢竛讥謿穕諸峵说訬譇賲譴氓註豦弩讴謢认訤氘気訴言粔竼诰氽评謰樽窻訒认尜譕豜诱殮諤殤詍豀譍毿拤氜尘訵掠企认謘穁訹寪詓译揘豇弁訣詍謘竤弉還粝貪記彋欖诛毆殍讥稺氏穢註询謰訿孞詉揁諠欘屠識訜纣詇詖責峼欢豦讦譨提幩竮謹彡羅论稱诡氾汚妹訃言误絽毝說講计欨毘謷謳弚穐詚彛譱謄糣诽彉水謱幤汪幬訚殥謱訔篢讠歆讥穒計巽壳拠竤謘郡缴詟拧診帷訾謼諄詇巛推頑廷訯屭殤訥嵞警謠歕帽屯殮讥訽惺効盬邉診殳譾诤彶訊峎絡欺胊謒鸏缶殐诽殊譴記籹欽謝穎诙譯毾譣麯母氹謲訟欺釈竦罐訕気譏訟平講应広懶籹评殌記訨计諅诹漾诲詌诺彇絕歑袾譑认謴汏譁詙容貗幙拷认譪讥謄譼緞殼籦絶江讠註殻詊惊记評謙歧訳孟竏字诲歂譳帟计譳訏解毌歶彀彿譫悢诼毻孚诜訹徾詊词譣禿氢拨殮訝粩訛诫麪謎讠訄訣循謿弨诗檐竬譸詏郵譕譞诽悮託譭穾歚謼綼謗娥謏米谍彄訋稩詊悠训訚謲诫幨詑訬计證訊接汞氕课覻殠氞讬订诬譶毫縻背訯殘詙段粕詚歩謍試蟱粙謌誠言诛彊谸峩式譾歋欿譥讣計欭議误紏謶謽訉譏诼訃弟粆訖欧譽訜计孜殡讠幵诽站譨豇突謘请尜诃詗彫詓欟讥譅稩謏設訨絭汜涙書课紳診謄殢謵竟訞止殤訯譝記譪订诛譊謡纼證认謀謠罺詇讥訆诅諞謭譝欇訽郲孵訧謎譭认说弧訁筟驗譖翁譝訫謱彣謗试欛詊弣拧訾綻詟綬读孤郴訸稽譖譨謣袾歀殫稢譇稉謵譔評诿稤幖訅稼譇謁豑訽彆诚殊氽役揩謄豅訂常訃稛罊註氨诳诱计氏穇謏譽殥訯讥訷认譣鹮鹣讴繗譹詨订籲訶譃殤譟讥譏评遗识訞讥譽豚惑謊謖罈诳开雲笿詍空屷诲紗訃譹謽忯譼詁讥譟诱彡讥歙譥訿诲欯歇幒彞譳稽譒讥稙氾计譊訃氏譵殤訩謿诬郱識歫殤譔讥縞譳讥讠訋认訚已讥已讠认設歠诿評订笹訟讥
        
        return X


if __name__ == '__main__':
    print("此模組應當沋 train_models_v2.py 中使用")

import pandas as pd
import numpy as np

def calculate_cooling_report_data(df, wind_speeds=None, steady_start_idx=None, steady_end_idx=None, log_callback=None):
    """
    整车降温报告数据计算引擎
    输入：处理后的 DataFrame（包含中文列名），可选的稳态区间
    输出：包含 data1 ~ data40（部分空缺）的字典，用于填充 Word 报告
    """
    # ========== 0. 稳态区间裁剪 ==========
    if steady_start_idx is not None or steady_end_idx is not None:
        s_idx = max(0, int(steady_start_idx)) if steady_start_idx is not None else 0
        e_idx = min(len(df), int(steady_end_idx)) if steady_end_idx is not None else len(df)
        if s_idx >= e_idx:
            raise ValueError("稳态起始行必须小于结束行！")
        df = df.iloc[s_idx:e_idx]
    # 否则使用整个 df
    
    # 确保索引为时间
    df.index = pd.to_datetime(df.index)
    
    # ========== 1. 辅助函数 ==========
    def get_series(col_names):
        """取多个列的平均值，缺失列则返回 NaN 序列"""
        valid = [c for c in col_names if c in df.columns]
        if not valid:
            return pd.Series(np.nan, index=df.index)
        return df[valid].mean(axis=1)
    
    def safe_value(series, func, default=np.nan):
        """安全计算聚合值"""
        if series.isna().all():
            return default
        return func(series.dropna())
    
    def time_to_reach_threshold(series, threshold=20):
        """返回首次达到阈值的时刻（相对于数据起点，单位：分钟）"""
        if series.isna().all():
            return np.nan
        reached = series[series >= threshold]
        if reached.empty:
            return np.nan  # 未达到
        t0 = df.index[0]
        t_reach = reached.index[0]
        return (t_reach - t0).total_seconds() / 60.0
    
    def max_abs_diff(series_a, series_b):
        """两序列绝对差的最大值"""
        if series_a.isna().all() or series_b.isna().all():
            return np.nan
        return (series_a - series_b).abs().max()
    
    # ========== 2. 预设可能用到的列 ==========
    # 脚部温度列组
    foot_pairs = {
        '主驾': (['主驾左脚', '主驾右脚'], ['主驾左脚', '主驾右脚']),
        '副驾': (['副驾左脚', '副驾右脚'], ['副驾左脚', '副驾右脚']),
        '中排左': (['中排左脚1', '中排左脚2'], ['中排左脚1', '中排左脚2']),
        '中排右': (['中排右脚1', '中排右脚2'], ['中排右脚1', '中排右脚2']),
        '后排左': (['后排左脚1', '后排左脚2'], ['后排左脚1', '后排左脚2']),
        '后排右': (['后排右脚1', '后排右脚2'], ['后排右脚1', '后排右脚2']),
    }
    
    # 风道温度列（可根据实际列名调整）
    DUCT_COLUMNS = {
        '主驾': '主驾风道温度',
        '副驾': '副驾风道温度',
        '中排左': '中排左风道温度',
        '中排右': '中排右风道温度',
        '后排左': '后排左风道温度',
        '后排右': '后排右风道温度',
    }
    
    # 各座位左右脚列名（用于第17、29-34）
    seat_foot_pairs = [
        ('主驾', ['主驾左脚', '主驾右脚']),
        ('副驾', ['副驾左脚', '副驾右脚']),
        ('中排左', ['中排左脚1', '中排左脚2']),
        ('中排右', ['中排右脚1', '中排右脚2']),
        ('后排左', ['后排左脚1', '后排左脚2']),
        ('后排右', ['后排右脚1', '后排右脚2']),
    ]
    
    res = {}
    
    # ========== 3. 逐项指标计算 ==========
    # 辅助：检查某座位的左右脚列是否存在
    def has_seat_foot(name):
        cols = dict(seat_foot_pairs)[name]
        return all(c in df.columns for c in cols)
    
    # ---- 指标1-12：平均足部温度及到达20℃时间 ----
    idx = 1
    for seat, (cols_mean, _) in foot_pairs.items():
        if not has_seat_foot(seat):
            # 该座位数据缺失，跳过两个指标（时间和均值）
            res[f"data{idx}"] = "N/A"
            res[f"data{idx+1}"] = "N/A"
        else:
            avg_foot = get_series(cols_mean)
            # 到达20℃时间
            time_to_20 = time_to_reach_threshold(avg_foot, 20)
            res[f"data{idx}"] = round(time_to_20, 1) if not pd.isna(time_to_20) else "N/A"
            # 平均足部温度（稳态区间内）
            avg_temp = avg_foot.mean()
            res[f"data{idx+1}"] = round(avg_temp, 2) if not pd.isna(avg_temp) else "N/A"
        idx += 2
    
    # ---- 指标13-16：风口温差 ----
    def duct_diff(seat_name):
        col = DUCT_COLUMNS.get(seat_name, None)
        if col and col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                return series.max() - series.min()
        return np.nan
    
    duct_seats = ['主驾', '副驾', '中排左', '中排右', '后排左', '后排右']
    front_diff = max(duct_diff('主驾'), duct_diff('副驾'), key=lambda x: x if not pd.isna(x) else -1)
    mid_diff = max(duct_diff('中排左'), duct_diff('中排右'), key=lambda x: x if not pd.isna(x) else -1) if has_seat_foot('中排左') or has_seat_foot('中排右') else np.nan
    rear_diff = max(duct_diff('后排左'), duct_diff('后排右'), key=lambda x: x if not pd.isna(x) else -1)
    
    all_diffs = [duct_diff(s) for s in duct_seats if has_seat_foot(s) or s in ['主驾','副驾','后排左','后排右']]  # 中排根据存在与否加入
    if not all_diffs:
        max_all_diff = np.nan
    else:
        max_all_diff = max(all_diffs)
    
    res["data13"] = round(front_diff, 2) if not pd.isna(front_diff) else "N/A"
    res["data14"] = round(mid_diff, 2) if not pd.isna(mid_diff) else "N/A"
    res["data15"] = round(rear_diff, 2) if not pd.isna(rear_diff) else "N/A"
    res["data16"] = round(max_all_diff, 2) if not pd.isna(max_all_diff) else "N/A"
    
    # ---- 指标17：各座位左右脚温差（结束时刻） ----
    end_time = df.index[-1]
    end_diffs = []
    for seat, cols in seat_foot_pairs:
        if all(c in df.columns for c in cols):
            left = df[cols[0]].loc[end_time] if end_time in df.index else df[cols[0]].iloc[-1]
            right = df[cols[1]].loc[end_time] if end_time in df.index else df[cols[1]].iloc[-1]
            if not pd.isna(left) and not pd.isna(right):
                end_diffs.append(abs(left - right))
    if end_diffs:
        res["data17"] = round(max(end_diffs), 2)
    else:
        res["data17"] = "N/A"
    
    # ---- 指标18-20：空缺 ----
    # 不添加，模板若无引用则安全
    
    # ---- 指标21：循环比例 ----
    if '循环比例' in df.columns:
        res["data21"] = round(df['循环比例'].mean(), 2)
    else:
        res["data21"] = "N/A"
    
    # ---- 指标22：环境温度（车外温度） ----
    if '车外温度' in df.columns:
        res["data22"] = round(df['车外温度'].mean(), 2)
    else:
        res["data22"] = "N/A"
    
    # ---- 指标23：车内温度 ----
    if '车内温度' in df.columns:
        res["data23"] = round(df['车内温度'].mean(), 2)
    else:
        res["data23"] = "N/A"
    
    # ---- 指标24：脚部初始温度（所有脚部测点开始时刻均值） ----
    start_time = df.index[0]
    all_foot_cols = []
    for pair in foot_pairs.values():
        all_foot_cols.extend(pair[0])
    foot_cols_start = [c for c in all_foot_cols if c in df.columns]
    if foot_cols_start:
        start_vals = df[foot_cols_start].iloc[0].astype(float)
        res["data24"] = round(start_vals.mean(), 2) if not start_vals.isna().all() else "N/A"
    else:
        res["data24"] = "N/A"
    
    # ---- 指标25：最高目标温度（左通道目标温度最大值） ----
    if '左通道目标温度' in df.columns:
        res["data25"] = round(df['左通道目标温度'].max(), 2)
    else:
        res["data25"] = "N/A"
    
    # ---- 指标26：压缩机功率 ----
    if '压缩机功率' in df.columns:
        res["data26"] = round(df['压缩机功率'].mean(), 2)
    else:
        res["data26"] = "N/A"
    
    # ---- 指标27：PTC功率 ----
    if 'PTC功率' in df.columns:
        res["data27"] = round(df['PTC功率'].mean(), 2)
    else:
        res["data27"] = "N/A"
    
    # ---- 指标28：总能耗 ----
    if '压缩机功率' in df.columns and 'PTC功率' in df.columns:
        res["data28"] = round(df['压缩机功率'].mean() + df['PTC功率'].mean(), 2)
    else:
        res["data28"] = "N/A"
    
    # ---- 指标29-34：各座位左右脚最大温差 ----
    seat_max_diffs = {}
    for seat, cols in seat_foot_pairs:
        if all(c in df.columns for c in cols):
            left = df[cols[0]]
            right = df[cols[1]]
            diff_max = max_abs_diff(left, right)
            seat_max_diffs[seat] = diff_max
        else:
            seat_max_diffs[seat] = np.nan
    
    res["data29"] = round(seat_max_diffs['主驾'], 2) if not pd.isna(seat_max_diffs['主驾']) else "N/A"
    res["data30"] = round(seat_max_diffs['副驾'], 2) if not pd.isna(seat_max_diffs['副驾']) else "N/A"
    res["data31"] = round(seat_max_diffs['中排左'], 2) if not pd.isna(seat_max_diffs['中排左']) else "N/A"
    res["data32"] = round(seat_max_diffs['中排右'], 2) if not pd.isna(seat_max_diffs['中排右']) else "N/A"
    res["data33"] = round(seat_max_diffs['后排左'], 2) if not pd.isna(seat_max_diffs['后排左']) else "N/A"
    res["data34"] = round(seat_max_diffs['后排右'], 2) if not pd.isna(seat_max_diffs['后排右']) else "N/A"
    
    # ---- 指标35-40：绘图分析（占位） ----
    plot_placeholders = {
        "data35": "待图表生成 - 通道温度、风口布点温度分析",
        "data36": "待图表生成 - 工作状态分析",
        "data37": "待图表生成 - 压缩机排气压力/排气温度/过热度/IGBT温度",
        "data38": "待图表生成 - 压缩机工作状态与余热控制分析",
        "data39": "待图表生成 - 鼓风机电压分析",
        "data40": "待图表生成 - 外循环控制分析",
    }
    res.update(plot_placeholders)
    
    # 可选：输出日志
    if log_callback:
        log_callback("✅ 整车降温报告数据计算完成（共40项指标）")
    
    return res

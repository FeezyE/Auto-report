import pandas as pd
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from 查表参数 import df_target_temp, df_target_temp_F, tables, tables_foot, df_ramp_rate, df_wind_max
    HAS_TABLES = True
except ImportError as e:
    HAS_TABLES = False
    print(f"⚠️ 无法导入完整的查表参数: {e}")

def calculate_comfort_report_data(df, wind_speeds=None, steady_start_idx=None, steady_end_idx=None, log_callback=None, target_score=6.5):
    """
    舒适性稳态判定及报告数据计算引擎
    说明：data1/data2 要求文本跟随 target_score，但得分计算 data5/data6 维持原逻辑不受影响。
    """
    if not HAS_TABLES:
        raise ValueError("缺少查表文件，无法计算打分数据！请确保查表参数.py无误。")

    df.index = pd.to_datetime(df.index)
    
    # 确定设定温度的参照列
    set_col = '对应亚洲设定温度' if '对应亚洲设定温度' in df.columns else '主驾设定温度'
    
    df = df.dropna(subset=['车外温度', set_col])
    if df.empty:
        raise ValueError("错误: 有效判定数据（外温/设定）全为空！")

    last_amb_temp = df['车外温度'].iloc[-1]
    is_winter = last_amb_temp <= 15
    season = 'winter' if is_winter else 'summer'

    def get_series(col_names):
        valid_cols = [c for c in col_names if c in df.columns]
        if not valid_cols:
            return pd.Series(np.nan, index=df.index)
        return df[valid_cols].mean(axis=1)

    face_dr = get_series(['主驾左脸', '主驾右脸'])
    foot_dr = get_series(['主驾左脚', '主驾右脚'])
    face_pa = get_series(['副驾左脸', '副驾右脸'])
    foot_pa = get_series(['副驾左脚', '副驾右脚'])
    face_rl = get_series(['后排左脸']) 
    foot_rl = get_series(['后排左脚1', '后排左脚2'])
    face_rr = get_series(['后排右脸']) 
    foot_rr = get_series(['后排右脚1', '后排右脚2'])
    
    blower_volt = df.get('鼓风机电压', pd.Series(0, index=df.index))

    if '实际风速' in df.columns:
        blower_speed = df['实际风速']
    elif '鼓风机挡位' in df.columns and wind_speeds is not None and HAS_SCIPY:
        x_pts = [0, 10, 20, 30, 40, 50, 60, 70]
        if len(wind_speeds) == 8:
            pchip = PchipInterpolator(x_pts, wind_speeds)
            clipped_lvl = df['鼓风机挡位'].fillna(0).clip(0, 70)
            blower_speed = pd.Series(pchip(clipped_lvl).clip(min=0), index=df.index)
        else:
            blower_speed = df['鼓风机挡位']
    else:
        blower_speed = df.get('鼓风机挡位', pd.Series(0, index=df.index))

    def find_steady_state_start(series):
        if series.isna().all(): return df.index[-1]
        rolling_max = series.rolling(window=60, min_periods=60).max()
        rolling_min = series.rolling(window=60, min_periods=60).min()
        fluctuation = rolling_max - rolling_min
        steady_points = fluctuation[fluctuation < 0.2]
        start_time = df.index[0]
        max_time_limit = start_time + pd.Timedelta(minutes=30) 
        if not steady_points.empty:
            steady_start = steady_points.index[0]
            if steady_start <= max_time_limit: 
                return steady_start
        if df.index[-1] > max_time_limit: 
            return max_time_limit
        return df.index[-1] 

    def get_steady_df(series=None, min_start_time=None):
        if steady_start_idx is not None:
            s_idx = max(0, int(steady_start_idx))
            e_idx = min(len(df), int(steady_end_idx)) if steady_end_idx is not None else len(df)
            if s_idx < e_idx: 
                return df.index[s_idx], df.iloc[s_idx:e_idx]
                
        start_t = find_steady_state_start(series)
        if min_start_time is not None and start_t < min_start_time: 
            start_t = min_start_time
            
        return start_t, df.loc[start_t:]

    def score_by_limits(val, limit_list):
        if pd.isna(val): return 0.0
        scores = [10, 9, 8, 7, 6]
        if val > limit_list[-1]: return 0.0
        if val <= limit_list[0]: return 10.0
        return float(np.interp(val, limit_list, scores))

    def get_ambient_row_limits(df_table, amb):
        amb_idx = max(-35, min(50, int(round(amb))))
        if amb_idx not in df_table.index:
            amb_idx = min(df_table.index, key=lambda x: abs(x - amb_idx))
        row = df_table.loc[amb_idx]
        limits = row.iloc[0:5].values.tolist()
        clean_limits = [float(x) if pd.notna(x) and x is not None else 999.0 for x in limits]
        return clean_limits

    # --- 动态区间文本获取 (仅改变 data1 / data2 的呈现) ---
    def get_requirement_text(center, off_dict, score, season, list_idx):
        def get_val_by_key(k, side):
            arr = off_dict[season][side].get(k, off_dict[season][side].get(str(k)))
            return abs(arr[min(list_idx, len(arr)-1)])

        if score == 6.0:
            up_val = get_val_by_key(6, 'upper')
            dn_val = get_val_by_key(6, 'lower')
        elif score == 7.0:
            up_val = get_val_by_key(7, 'upper')
            dn_val = get_val_by_key(7, 'lower')
        else:
            up_val = (get_val_by_key(6, 'upper') + get_val_by_key(7, 'upper')) / 2.0
            dn_val = (get_val_by_key(6, 'lower') + get_val_by_key(7, 'lower')) / 2.0
            
        txt = f"{center-dn_val:.1f} ~ {center+up_val:.1f}"
        return txt, center-dn_val, center+up_val, up_val, dn_val

    # --- 原汁原味的全局评分逻辑 (data5 / data6，不受 target_score 影响) ---
    def get_temp_mean_score(val, center, off_dict, season, list_idx):
        if pd.isna(val) or pd.isna(center): return 0.0
        diff = val - center
        
        def get_vals(key):
            res = []
            for sc in [10, 9, 8, 7, 6]:
                arr = off_dict[season][key].get(sc, off_dict[season][key].get(str(sc)))
                safe_i = min(list_idx, len(arr)-1)
                res.append(abs(arr[safe_i])) 
            return res
        
        up_limits = get_vals('upper')
        dn_limits = get_vals('lower')
        
        if diff >= 0: 
            return score_by_limits(diff, up_limits)
        else: 
            return score_by_limits(abs(diff), dn_limits)

    def calculate_seat_metrics(face_s, foot_s, is_dr=True, dict_base=0, seat_name="", min_start_time=None):
        ctrl_series = foot_s if is_winter else face_s
        steady_start_time, steady_df = get_steady_df(ctrl_series, min_start_time)
        
        steady_face = face_s.loc[steady_df.index]
        steady_foot = foot_s.loc[steady_df.index]
        
        mean_amb = steady_df['车外温度'].mean() if not steady_df.empty else 20
        mean_set = steady_df[set_col].mean() if not steady_df.empty else 22
        
        amb_idx = max(-30, min(44, int(round(mean_amb))))
        st_idx = max(17, min(35, int(round(mean_set))))
        list_idx = st_idx - 17
        
        if log_callback:
            start_row = df.index.get_loc(steady_df.index[0]) + 1
            end_row = df.index.get_loc(steady_df.index[-1]) + 1
            log_callback(f"    📍 [{seat_name}] 取稳态: 第 {start_row} 行 到 第 {end_row} 行 -> 基准: 外温 {amb_idx}℃, 设定 {st_idx}℃")
            
        try:
            c_face = df_target_temp.loc[amb_idx, st_idx]
            c_foot = df_target_temp_F.loc[amb_idx, st_idx]
        except:
            c_face, c_foot = 25.0, 28.0

        # 获取动态要求区间 (受 target_score 影响，用于在报告里展示文本)
        face_txt, f_dn, f_up, f_up_off, f_dn_off = get_requirement_text(c_face, tables, target_score, season, list_idx)
        foot_txt, ft_dn, ft_up, ft_up_off, ft_dn_off = get_requirement_text(c_foot, tables_foot, target_score, season, list_idx)
        
        # 详尽的日志打印
        debug_msg = (
            f"      🔍 [核对 - {seat_name}稳态要求] 展现文本基于 {target_score}分线:\n"
            f"         1. 面部(data1): 目标中心点={c_face}℃, 查表下限偏移={f_dn_off}, 上限偏移={f_up_off} => 最终区间: {face_txt}\n"
            f"         2. 脚部(data2): 目标中心点={c_foot}℃, 查表下限偏移={ft_dn_off}, 上限偏移={ft_up_off} => 最终区间: {foot_txt}"
        )
        if log_callback:
            log_callback(debug_msg)
        else:
            print(debug_msg)
        
        data3 = steady_face.mean()
        data4 = steady_foot.mean()
        
        # 实际得分计算 (完全按照原先逻辑，不受 target_score 影响)
        data5 = get_temp_mean_score(data3, c_face, tables, season, list_idx)
        data6 = get_temp_mean_score(data4, c_foot, tables_foot, season, list_idx)
        
        start_time = df.index[0]
        
        if is_winter:
            reached = foot_s[(foot_s >= ft_dn) & (foot_s <= ft_up)]
        else:
            reached = face_s[(face_s >= f_dn) & (face_s <= f_up)]
            
        if reached.empty:
            if is_winter:
                reached = foot_s[foot_s >= ft_dn]
            else:
                reached = face_s[face_s <= f_up]
        
        if not reached.empty:
            data7 = (reached.index[0] - start_time).total_seconds() / 60.0
            ramp_limits = get_ambient_row_limits(df_ramp_rate, mean_amb)
            data8 = ramp_limits[3]
            data9 = score_by_limits(data7, ramp_limits)
        else:
            data7 = 999.9  
            ramp_limits = get_ambient_row_limits(df_ramp_rate, mean_amb)
            data8 = ramp_limits[3]
            data9 = 0.0
            
        data10 = blower_speed.loc[steady_df.index].max() if not steady_df.empty else 0.0
        blower_limits = get_ambient_row_limits(df_wind_max, mean_amb)
        data11 = blower_limits[3] 
        data12 = score_by_limits(data10, blower_limits)
        
        steady_v = blower_volt.loc[steady_df.index]
        volt_diff = steady_v.max() - steady_v.min() if not steady_v.empty else 0.0
        volt_score = score_by_limits(volt_diff, [0.2, 0.4, 0.6, 0.8, 1.0])
        
        face_fluc = steady_face.max() - steady_face.min()
        face_fluc_score = score_by_limits(face_fluc, [0.5, 1.0, 1.5, 2.0, 3.0])
        face_fluc_limit = 2.0  
        
        foot_fluc = steady_foot.max() - steady_foot.min()
        foot_fluc_score = score_by_limits(foot_fluc, [0.5, 1.0, 2.0, 3.0, 4.0])
        foot_fluc_limit = 3.0  
        
        if is_winter:
            overall_score = (0.20 * data5 + 0.30 * data6 + 0.20 * data9 +
                             0.10 * data12 + 0.10 * volt_score +
                             0.05 * face_fluc_score + 0.05 * foot_fluc_score)
        else:
            overall_score = (0.50 * data5 + 0.20 * data9 + 0.10 * data12 +
                             0.10 * volt_score + 0.10 * face_fluc_score)

        res = {}
        res[f"data{1+dict_base}"] = face_txt
        res[f"data{2+dict_base}"] = foot_txt
        res[f"data{3+dict_base}"] = round(data3, 2) if not pd.isna(data3) else "N/A"
        res[f"data{4+dict_base}"] = round(data4, 2) if not pd.isna(data4) else "N/A"
        res[f"data{5+dict_base}"] = round(data5, 2)
        res[f"data{6+dict_base}"] = round(data6, 2)
        res[f"data{7+dict_base}"] = round(data7, 1) if data7 != 999.9 else "未达标"
        res[f"data{8+dict_base}"] = str(data8)
        res[f"data{9+dict_base}"] = round(data9, 2)
        res[f"data{10+dict_base}"] = str(round(data10, 2))
        res[f"data{11+dict_base}"] = str(data11)
        res[f"data{12+dict_base}"] = round(data12, 2)
        
        offset = 0
        if is_dr:
            res[f"data{13+dict_base}"] = round(volt_diff, 2)
            res[f"data{14+dict_base}"] = round(volt_score, 2)
        else:
            offset = -2 
            
        res[f"data{15+offset+dict_base}"] = round(face_fluc, 2) if not pd.isna(face_fluc) else "N/A"
        res[f"data{16+offset+dict_base}"] = round(face_fluc_score, 2)
        res[f"data{17+offset+dict_base}"] = str(face_fluc_limit)
        res[f"data{18+offset+dict_base}"] = round(foot_fluc, 2) if not pd.isna(foot_fluc) else "N/A"
        res[f"data{19+offset+dict_base}"] = round(foot_fluc_score, 2)
        res[f"data{20+offset+dict_base}"] = str(foot_fluc_limit)
        res[f"data{21+offset+dict_base}"] = round(overall_score, 2)
        
        return res, steady_start_time

    final_report = {}
    if log_callback: log_callback("\n📏 开始执行报告指标结算...")
    
    res_dr, dr_start_time = calculate_seat_metrics(face_dr, foot_dr, is_dr=True, dict_base=0, seat_name="主驾")
    final_report.update(res_dr)
    
    res_pa, _ = calculate_seat_metrics(face_pa, foot_pa, is_dr=False, dict_base=21, seat_name="副驾", min_start_time=dr_start_time)
    final_report.update(res_pa)
    
    res_rl, _ = calculate_seat_metrics(face_rl, foot_rl, is_dr=False, dict_base=40, seat_name="左后", min_start_time=dr_start_time)
    final_report.update(res_rl)
    
    res_rr, _ = calculate_seat_metrics(face_rr, foot_rr, is_dr=False, dict_base=59, seat_name="右后", min_start_time=dr_start_time)
    final_report.update(res_rr)
    
    _, global_steady_df = get_steady_df(foot_dr if is_winter else face_dr, min_start_time=dr_start_time)
    
    def safe_round(val):
        return int(round(val)) if not pd.isna(val) else "N/A"

    final_report["data79"] = safe_round(global_steady_df.get('车外温度', pd.Series(dtype=float)).mean())
    final_report["data80"] = safe_round(global_steady_df.get('压缩机转速', pd.Series(dtype=float)).mean())
    final_report["data81"] = safe_round(global_steady_df.get('压缩机功率', pd.Series(dtype=float)).mean())
    
    recirc = global_steady_df.get('循环比例', pd.Series(dtype=float))
    final_report["data82"] = safe_round((recirc.max() + recirc.min()) / 2) if not recirc.isna().all() else "N/A"
    
    defrost = global_steady_df.get('除霜比例', pd.Series(dtype=float))
    final_report["data83"] = safe_round((defrost.max() + defrost.min()) / 2) if not defrost.isna().all() else "N/A"
    
    l_solar = global_steady_df.get('左侧阳光值', pd.Series(dtype=float)).mean()
    r_solar = global_steady_df.get('右侧阳光值', pd.Series(dtype=float)).mean()
    
    if pd.isna(l_solar) and pd.isna(r_solar): 
        solar_val = np.nan
    elif pd.isna(r_solar) or r_solar == 0: 
        solar_val = l_solar if not pd.isna(l_solar) else 0
    elif pd.isna(l_solar) or l_solar == 0: 
        solar_val = r_solar if not pd.isna(r_solar) else 0
    else: 
        solar_val = (l_solar + r_solar) / 2
        
    final_report["data84"] = safe_round(solar_val)

    # ==============================================================
    # 🌟 新增高阶综合热管理指标 (位于 data85 及之后)
    # ==============================================================
    
    # 预留占位符 (PTC与玻璃湿度)
    final_report["data85"] = "待后续计算"  # PTC功率
    final_report["data86"] = "待后续计算"  # 玻璃湿度
    
    # 1. 余热利用状态分类判定 (data87) -> 文字占比法
    wh_status_series = global_steady_df.get('余热利用状态', pd.Series(dtype=object))
    if not wh_status_series.isna().all():
        # 计算各状态出现的频率占比
        counts = wh_status_series.value_counts(normalize=True)
        top_state = counts.index[0] # 占比最高的状态名
        top_pct = counts.iloc[0]    # 对应的比例 (0.0~1.0)
        
        if top_pct > 0.9:
            final_report["data87"] = f"全程{top_state}"
        else:
            final_report["data87"] = "余热补偿与余热利用交替"
    else:
        final_report["data87"] = "N/A"

    # 2. 余热补偿目标水温均值 (data88)
    wh_target_temp = global_steady_df.get('余热补偿目标水温', pd.Series(dtype=float)).mean()
    final_report["data88"] = round(wh_target_temp, 2) if not pd.isna(wh_target_temp) else "N/A"

    # 3. 三通水阀位置 (data89) -> ⚠️请在这里直接替换字典左侧的乱码文字
    valve3_series = global_steady_df.get('三通水阀位置', pd.Series(dtype=object))
    if not valve3_series.isna().all():
        # 取占比最高的状态
        v3_counts = valve3_series.value_counts()
        v3_val = v3_counts.index[0] if not v3_counts.empty else "N/A"
        # 映射字典
        valve3_map = {
            "乱码1": "无效",
            "乱码2": "连通电机散热器",
            "乱码3": "短接电机散热器",
            "乱码4": "中间位置"
        }
        final_report["data89"] = valve3_map.get(str(v3_val), str(v3_val))
    else:
        final_report["data89"] = "N/A"

    # 4. 四通水阀模式 (data90)
    # 原始列名可能包含字符串（如“大循环”），需要先映射成数字
    valve4_series_raw = global_steady_df.get('四通水阀状态（域控4.0）', pd.Series(dtype=object))
    if not valve4_series_raw.isna().all():
        # 定义两个映射表
        # ① 原始字符串 → 数值
        str_to_valve_num = {
            "无效": 0,
            "散热器小循环": 1,
            "大循环": 2,
            "板换小循环": 3,
            "中间位置": 4,
            # 如果数据中还有别的字符串，可在此补充
        }
        # ② 数值 → 描述文字（与原始映射保持一致）
        valve_num_to_desc = {
            0: "无效",
            1: "散热器小循环",
            2: "大循环",
            3: "板换小循环",
            4: "中间位置",
        }
        
        # 将列统一转换为数值
        numeric_vals = []
        for v in valve4_series_raw:
            if pd.isna(v):
                continue
            if isinstance(v, (int, float)):
                # 已经是数值，直接使用（但要确保在 0~4 范围内）
                numeric_vals.append(v)
            else:
                # 字符串类型，尝试查找映射表
                str_v = str(v).strip()
                if str_v in str_to_valve_num:
                    numeric_vals.append(str_to_valve_num[str_v])
                else:
                    # 未知字符串，记录警告并跳过（或赋默认值）
                    print(f"警告: 未知的四通水阀状态 '{str_v}'，已忽略")
        
        if numeric_vals:
            # 对转换后的数值求平均
            v4_mean = np.mean(numeric_vals)
            # 将平均数值取整（四舍五入）得到最接近的模式
            v4_mode_int = int(round(v4_mean))
            # 映射回描述文字
            desc = valve_num_to_desc.get(v4_mode_int, f"未知模式({v4_mode_int})")
            # 可选：在备注中保留平均数值（方便调试）
            final_report["data90"] = f"四通水阀模式: {desc} (均值={v4_mean:.2f})"
        else:
            final_report["data90"] = "N/A"
    else:
        final_report["data90"] = "N/A"

    # 5. 左吹面通道温度与蒸发器温度最大差值 (data91)
    t_left_face = global_steady_df.get('左吹面通道温度', pd.Series(dtype=float))
    t_evap = global_steady_df.get('蒸发器温度', pd.Series(dtype=float))
    if not t_left_face.isna().all() and not t_evap.isna().all():
        diff_max_evap = (t_left_face - t_evap).max()
        final_report["data91"] = round(diff_max_evap, 2)
    else:
        final_report["data91"] = "N/A"

    # 6. 左通道目标温度与左吹面通道温度绝对最大差值 (data92)
    t_left_target = global_steady_df.get('左通道目标温度', pd.Series(dtype=float))
    if not t_left_target.isna().all() and not t_left_face.isna().all():
        diff_max_target = (t_left_target - t_left_face).abs().max()
        final_report["data92"] = round(diff_max_target, 2)
    else:
        final_report["data92"] = "N/A"

    # 7. 蒸发器出口压力均值 与 显著波动判定 (data93, data94)
    p_evap_out = global_steady_df.get('蒸发器出口压力', pd.Series(dtype=float))
    if not p_evap_out.isna().all():
        p_mean = p_evap_out.mean()
        p_max = p_evap_out.max()
        final_report["data93"] = round(p_mean, 2)
        # 判断最大值与均值差距是否 > 0.1
        final_report["data94"] = "存在显著波动" if (p_max - p_mean) > 0.1 else "无显著波动现象"
    else:
        final_report["data93"] = "N/A"
        final_report["data94"] = "N/A"

    # 8. 蒸发器出口过热度均值 (data95)
    sh_evap_out = global_steady_df.get('蒸发器出口过热度', pd.Series(dtype=float)).mean()
    final_report["data95"] = round(sh_evap_out, 2) if not pd.isna(sh_evap_out) else "N/A"

    return final_report

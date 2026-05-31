#!/usr/bin/env python3
# KLineQualityDetector 回测框架 v1.1
# ETH 15分钟周期，支持参数扫描

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

CONFIG = {
    'symbol': 'ETH-USD',
    'timeframe': '15m',
    'days': 30,
    'vol_lookback': 20,
    'doji_threshold': 0.05,
    'small_body_threshold': 0.20,
    'max_shadow_to_body': 3.0,
    'max_wick_to_range': 0.70,
    'spike_body_multiplier': 4.0,
    'weight_body': 35,
    'weight_shadow_ctrl': 30,
    'weight_close_pos': 20,
    'weight_volume': 10,
    'weight_consistency': 5,
    'cluster_len': 3,
    'quality_threshold': 55,
    'avg_quality_threshold': 60,
    'body_growth_threshold': 0.8,
    'close_push_threshold': 1.005,
    'check_periods': {'6h': 24, '12h': 48, '24h': 96},
    'atr_expand_threshold': 1.5,
}

def fetch_data(symbol, timeframe, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start, end=end, interval=timeframe, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [c.lower() for c in df.columns]
    df = df[df['volume'] > 0]
    return df

def calculate_kline_quality(df, cfg):
    o, h, l, c, v = df['open'].values, df['high'].values, df['low'].values, df['close'].values, df['volume'].values
    rng = np.maximum(h - l, 0.001)
    body = np.abs(c - o)
    body_ratio = body / rng
    close_pos = (c - l) / rng
    us = (h - np.maximum(c, o)) / rng
    ls = (np.minimum(c, o) - l) / rng
    ts = us + ls
    stb = np.where(body > 0, ts * rng / body, 999)
    vsma = pd.Series(v).rolling(cfg['vol_lookback']).mean().values
    vr = np.where(vsma > 0, v / vsma, 1)
    atr14 = pd.Series(h - l).rolling(14).mean().values
    body_vs_atr = np.where(atr14 > 0, body / atr14, 1)
    is_doji = body_ratio < cfg['doji_threshold']
    is_small = (body_ratio < cfg['small_body_threshold']) & (~is_doji)
    is_ls = stb > cfg['max_shadow_to_body']
    is_spike = body_vs_atr > cfg['spike_body_multiplier']
    is_battle = (us > 0.2) & (ls > 0.2) & (body_ratio < 0.4)
    bs = body_ratio * cfg['weight_body']
    ss = (1 - np.minimum(ts, 1)) * cfg['weight_shadow_ctrl']
    ps_bull = close_pos * cfg['weight_close_pos']
    ps_bear = (1 - close_pos) * cfg['weight_close_pos']
    vs = np.minimum(vr, 2) * cfg['weight_volume']
    cb = np.where((ts < 0.2) & (body_ratio > 0.5), cfg['weight_consistency'], 0)
    pen = (is_doji * 40 + is_small * 25 + is_ls * 30 + is_spike * 35 + is_battle * 25)
    bull_s = np.clip(bs + ss + ps_bull + vs + cb - pen, 0, 100)
    bear_s = np.clip(bs + ss + ps_bear + vs + cb - pen, 0, 100)
    df['quality_score'] = np.where(c > o, bull_s, np.where(c < o, bear_s, 0))
    df['body_ratio'] = body_ratio
    df['total_shadow'] = ts
    df['shadow_to_body'] = stb
    df['is_doji'] = is_doji
    df['is_spike'] = is_spike
    df['is_long_shadow'] = is_ls
    return df

def detect_clusters(df, qt, at, gt, pt, cl):
    n = len(df)
    bs, rs = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    is_bull = (df['close'] > df['open']) & (df['quality_score'] >= qt) & (~df['is_doji']) & (~df['is_spike'])
    is_bear = (df['close'] < df['open']) & (df['quality_score'] >= qt) & (~df['is_doji']) & (~df['is_spike'])
    for i in range(1, n):
        if is_bull.iloc[i]: bs[i] = bs[i-1] + 1
        if is_bear.iloc[i]: rs[i] = rs[i-1] + 1
    body = np.abs(df['close'] - df['open']).values
    ib, ir = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for i in range(n):
        if bs[i] >= cl:
            s = i - bs[i] + 1
            if s >= 0:
                bg = body[i] / body[s] if body[s] > 0 else 1
                cp = df['close'].iloc[i] / df['close'].iloc[s] if df['close'].iloc[s] > 0 else 1
                aq = df['quality_score'].iloc[max(0,i-cl+1):i+1].mean()
                if aq >= at and bg >= gt and cp >= pt: ib[i] = True
        if rs[i] >= cl:
            s = i - rs[i] + 1
            if s >= 0:
                bg = body[i] / body[s] if body[s] > 0 else 1
                cp = df['close'].iloc[i] / df['close'].iloc[s] if df['close'].iloc[s] > 0 else 1
                aq = df['quality_score'].iloc[max(0,i-cl+1):i+1].mean()
                if aq >= at and bg >= gt and cp <= (1/pt): ir[i] = True
    return ib, ir, bs, rs

def run_backtest(df, ib, ir, cfg):
    n = len(df)
    atr = pd.Series(df['high'] - df['low']).rolling(14).mean().values
    cps, at = cfg['check_periods'], cfg['atr_expand_threshold']
    bi, ri = df.index[ib], df.index[ir]
    s = {}
    for d, sigs in [('bull', bi), ('bear', ri)]:
        for pn, pb in cps.items():
            ae, mp, ml = [], [], []
            for idx in sigs:
                i = df.index.get_loc(idx)
                if i + pb >= n: continue
                sp = df['close'].iloc[i]
                fh = df['high'].iloc[i+1:i+pb+1].max()
                fl = df['low'].iloc[i+1:i+pb+1].min()
                fa = atr[i+pb]
                if d == 'bull':
                    mp.append((fh - sp) / sp * 100)
                    ml.append((sp - fl) / sp * 100)
                else:
                    mp.append((sp - fl) / sp * 100)
                    ml.append((fh - sp) / sp * 100)
                ae.append(fa / atr[i] >= at)
            k = f'{d}_{pn}'
            s[f'{k}_count'] = len(mp)
            s[f'{k}_atr_rate'] = round(sum(ae)/len(ae)*100, 1) if ae else 0
            s[f'{k}_avg_profit'] = round(np.mean(mp), 2) if mp else 0
            s[f'{k}_avg_loss'] = round(np.mean(ml), 2) if ml else 0
            s[f'{k}_pl_ratio'] = round(np.mean(mp)/(np.mean(ml)+0.001), 2) if mp else 0
    s['bull_count'], s['bear_count'] = len(bi), len(ri)
    return s

def scan_params(df, cfg):
    results = []
    for cl in [2, 3]:
        for qt in [50, 55, 60]:
            for at in [55, 60, 65]:
                ib, ir, _, _ = detect_clusters(df, qt, at, cfg['body_growth_threshold'], cfg['close_push_threshold'], cl)
                s = run_backtest(df, ib, ir, cfg)
                results.append({
                    'cluster_len': cl, 'quality_thresh': qt, 'avg_quality_thresh': at,
                    'total': s['bull_count'] + s['bear_count'],
                    'bull_signals': s['bull_count'], 'bear_signals': s['bear_count'],
                    'atr_6h': (s.get('bull_6h_atr_rate',0) + s.get('bear_6h_atr_rate',0))/2,
                    'bull_6h_profit': s.get('bull_6h_avg_profit',0),
                    'bear_6h_profit': s.get('bear_6h_avg_profit',0),
                })
    return results

if __name__ == '__main__':
    df = fetch_data(CONFIG['symbol'], CONFIG['timeframe'], CONFIG['days'])
    df = calculate_kline_quality(df, CONFIG)
    scan = scan_params(df, CONFIG)
    print(f"数据: {len(df)} 根K线")
    print(f"扫描完成: {len(scan)} 种参数组合")
    df.to_csv('kqd_processed_data.csv')
    with open('kqd_scan_results.json', 'w') as f:
        json.dump(scan, f, indent=2)
    print("文件已保存")
"""
Detailed investigation of R^2 and momentum score behavior across:
1. Stationary white noise around flat price (pure micro-noise)
2. Mean-reverting Ornstein-Uhlenbeck oscillations (choppy sideways)
3. Spurious regression in un-drifted random walk vs deterministic trend
4. Multi-factor composite ranking behavior
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.momentum.clenow import compute_clenow_momentum, calculate_multi_timeframe_clenow
from src.momentum.volatility import calculate_jump_ratio_penalty, calculate_realized_volatility, calculate_downside_deviation
from src.momentum.classical import compute_jegadeesh_titman_momentum
from src.momentum.scoring import winsorized_z_score, compute_composite_momentum_score

np.random.seed(42)
n_days = 90
t = np.arange(n_days, dtype=np.float64)

print("=" * 80)
print("TEST 1: Stationary High-Frequency White Noise Around Flat Price (P = 100 + eps)")
print("=" * 80)
for noise_std in [0.1, 1.0, 5.0, 10.0]:
    r2_list, score_list = [], []
    for _ in range(500):
        p_white = 100.0 + np.random.normal(0, noise_std, n_days)
        p_white = np.clip(p_white, 1.0, None)
        slope, r2, score = compute_clenow_momentum(p_white, lookback=90)
        r2_list.append(r2)
        score_list.append(score)
    print(f"White Noise Std={noise_std:4.1f}: Mean R^2 = {np.mean(r2_list):.6f}, Mean Score = {np.mean(score_list):.6f}, Max R^2 = {np.max(r2_list):.6f}")

print("\n" + "=" * 80)
print("TEST 2: Choppy Cyclical / Sinusoidal Oscillations (Zero Trend, Pure Cycles)")
print("=" * 80)
# Sinusoidal cycle: P(t) = 100 + 10 * sin(2*pi*t / 20)
p_sine_20 = 100.0 + 10.0 * np.sin(2 * np.pi * t / 20.0)
slope_s20, r2_s20, score_s20 = compute_clenow_momentum(p_sine_20, lookback=90)
print(f"20-day Cycle (4.5 full cycles in 90d): Slope={slope_s20:.6f}, R^2={r2_s20:.6f}, Score={score_s20:.6f}")

p_sine_10 = 100.0 + 10.0 * np.sin(2 * np.pi * t / 10.0)
slope_s10, r2_s10, score_s10 = compute_clenow_momentum(p_sine_10, lookback=90)
print(f"10-day Cycle (9.0 full cycles in 90d): Slope={slope_s10:.6f}, R^2={r2_s10:.6f}, Score={score_s10:.6f}")

print("\n" + "=" * 80)
print("TEST 3: Mean-Reverting Ornstein-Uhlenbeck (Choppy Range-bound Market)")
print("=" * 80)
# dX_t = theta * (mu - X_t) dt + sigma dW_t
theta = 0.20  # strong mean reversion
mu = np.log(100.0)
ou_r2, ou_score = [], []
for _ in range(500):
    log_p = np.zeros(n_days)
    log_p[0] = mu
    for i in range(1, n_days):
        log_p[i] = log_p[i-1] + theta * (mu - log_p[i-1]) + np.random.normal(0, 0.02)
    p_ou = np.exp(log_p)
    sl, r2, sc = compute_clenow_momentum(p_ou, lookback=90)
    ou_r2.append(r2)
    ou_score.append(sc)
print(f"Ornstein-Uhlenbeck Mean-Reverting Market (500 runs):")
print(f"  Mean R^2:   {np.mean(ou_r2):.6f} (Theory: very low)")
print(f"  Median R^2: {np.median(ou_r2):.6f}")
print(f"  Mean Score: {np.mean(ou_score):.6f}")

print("\n" + "=" * 80)
print("TEST 4: Deterministic Trend vs Trend with Stationary Noise Overlay")
print("=" * 80)
# P(t) = 100 * exp(0.002 * t) + eps_t
for noise_std in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    r2_list, score_list = [], []
    for _ in range(200):
        p_trend_noisy = 100.0 * np.exp(0.002 * t) + np.random.normal(0, noise_std, n_days)
        p_trend_noisy = np.clip(p_trend_noisy, 1.0, None)
        sl, r2, sc = compute_clenow_momentum(p_trend_noisy, lookback=90)
        r2_list.append(r2)
        score_list.append(sc)
    print(f"Noise Std={noise_std:4.1f}: Mean Slope={np.mean(score_list)/np.mean(r2_list):.4f}, Mean R^2={np.mean(r2_list):.4f}, Mean Clenow Score={np.mean(score_list):.4f}")


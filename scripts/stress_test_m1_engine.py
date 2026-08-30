"""
Empirical Boundary Verification & Numerical Stability Stress-Testing Suite for M1 Engine.
Challenger 2 Verification Harness.
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.momentum.clenow import compute_clenow_momentum, calculate_multi_timeframe_clenow
from src.momentum.classical import compute_jegadeesh_titman_momentum, calculate_intermediate_momentum, calculate_multi_timeframe_returns
from src.momentum.volatility import (
    calculate_realized_volatility,
    calculate_downside_deviation,
    calculate_jump_ratio_penalty,
    calculate_atr,
    calculate_volatility_adjusted_momentum,
    compute_volatility_metrics,
)
from src.momentum.regime import (
    calculate_sma,
    calculate_ema,
    check_single_stock_trend_filter,
    check_golden_regime,
    calculate_52w_high_distance,
    evaluate_market_regime,
    evaluate_stock_eligibility,
)
from src.momentum.scoring import (
    winsorized_z_score,
    calculate_percentile_rank,
    compute_composite_momentum_score,
    rank_universe,
)
from src.momentum.models import MomentumMetrics, MarketRegimeState, RebalanceAction


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_extreme_price_scale_invariance() -> Dict[str, Any]:
    print_section("1. EXTREME PRICE SCALING & SCALE INVARIANCE")
    results = {}
    
    # Base price sequence: 253 days of smooth 0.1% daily compounding
    t = np.arange(253, dtype=np.float64)
    base_prices = 100.0 * np.power(1.001, t)
    
    scales = {
        "Micro Penny ($0.0001)": 1e-6,
        "Sub-Dollar Penny ($0.01)": 1e-4,
        "Low Price ($1.00)": 0.01,
        "Standard Price ($100.0)": 1.0,
        "High Price ($10,000.0)": 100.0,
        "Berkshire Scale ($1,000,000.0)": 10000.0,
        "Mega Scale ($100,000,000.0)": 1e6,
        "Astronomical ($10^12)": 1e10,
    }
    
    scale_metrics = []
    
    for name, scale in scales.items():
        p = base_prices * scale
        h = p * 1.01
        l = p * 0.99
        
        slope, r2, score = compute_clenow_momentum(p, lookback=90)
        mom12_1 = compute_jegadeesh_titman_momentum(p, lookback_total=252, skip_recent=21)
        mom6_1 = compute_jegadeesh_titman_momentum(p, lookback_total=126, skip_recent=21)
        vol_ann = calculate_realized_volatility(p, lookback=90)
        vol_down = calculate_downside_deviation(p, lookback=90)
        jump_ratio, jump_pen = calculate_jump_ratio_penalty(p, lookback=90)
        atr, natr = calculate_atr(h, l, p, period=14)
        
        scale_metrics.append({
            "name": name,
            "scale": scale,
            "start_p": p[0],
            "end_p": p[-1],
            "slope": slope,
            "r2": r2,
            "score": score,
            "mom12_1": mom12_1,
            "mom6_1": mom6_1,
            "vol_ann": vol_ann,
            "vol_down": vol_down,
            "jump_pen": jump_pen,
            "natr": natr,
        })
    
    df_scales = pd.DataFrame(scale_metrics)
    print(df_scales[["name", "start_p", "slope", "r2", "score", "mom12_1", "vol_ann", "natr"]].to_string(index=False))
    
    # Verification checks
    std_slope = df_scales["slope"].std()
    std_r2 = df_scales["r2"].std()
    std_score = df_scales["score"].std()
    std_mom12_1 = df_scales["mom12_1"].std()
    std_vol_ann = df_scales["vol_ann"].std()
    std_natr = df_scales["natr"].std()
    
    print("\nScale Invariance Precision (Std across all scales 10^-6 to 10^10):")
    print(f"  Slope Std:       {std_slope:.2e} (Expected: ~0)")
    print(f"  R^2 Std:         {std_r2:.2e} (Expected: ~0)")
    print(f"  Score Std:       {std_score:.2e} (Expected: ~0)")
    print(f"  Mom 12-1 Std:    {std_mom12_1:.2e} (Expected: ~0)")
    print(f"  Vol Ann Std:     {std_vol_ann:.2e} (Expected: ~0)")
    print(f"  NATR % Std:      {std_natr:.2e} (Expected: ~0)")
    
    scale_invariance_passed = (
        std_slope < 1e-10 and
        std_r2 < 1e-10 and
        std_score < 1e-10 and
        std_mom12_1 < 1e-10 and
        std_vol_ann < 1e-10 and
        std_natr < 1e-10
    )
    print(f"Scale Invariance Test Passed: {scale_invariance_passed}")
    results["scale_invariance_passed"] = scale_invariance_passed
    return results


def test_overflow_and_invalid_boundaries() -> Dict[str, Any]:
    print_section("2. OVERFLOW, UNDERFLOW, ZERO, AND INVALID BOUNDARY STRESS")
    results = {}
    
    # 2.1 Extreme Explosive Trend (e.g. 1000x gain in 90 days)
    t = np.arange(90, dtype=np.float64)
    p_explosive = 1.0 * np.exp(0.15 * t)  # beta = 0.15, scaled_beta = 37.5
    slope, r2, score = compute_clenow_momentum(p_explosive, lookback=90)
    print(f"Explosive Trend (beta=0.15): Slope={slope:,.2f}, R2={r2:.4f}, Score={score:,.2f} -> Finite: {math.isfinite(slope)}")
    
    # 2.2 Massive Overflow beta (beta = 5.0 -> scaled_beta = 1250 > 700)
    p_overflow = 1.0 * np.exp(5.0 * t)
    slope_ovf, r2_ovf, score_ovf = compute_clenow_momentum(p_overflow, lookback=90)
    print(f"Overflow Guard (beta=5.0): Slope={slope_ovf}, R2={r2_ovf:.4f}, Score={score_ovf} -> Guard Active: {math.isinf(slope_ovf)}")
    
    # 2.3 Extreme Collapse underflow (beta = -5.0 -> scaled_beta = -1250 < -700)
    p_underflow = 1000.0 * np.exp(-5.0 * t)
    slope_udf, r2_udf, score_udf = compute_clenow_momentum(p_underflow, lookback=90)
    print(f"Underflow Guard (beta=-5.0): Slope={slope_udf}, R2={r2_udf:.4f}, Score={score_udf} -> Guard Active: {slope_udf == -1.0}")
    
    # 2.4 Zeros, Negatives, and NaNs
    p_with_zero = np.array([10.0, 12.0, 0.0, 15.0] + [20.0]*86)
    slope_z, r2_z, score_z = compute_clenow_momentum(p_with_zero, lookback=90)
    print(f"Price with 0.0: ({slope_z}, {r2_z}, {score_z}) -> All NaN: {np.isnan(slope_z) and np.isnan(r2_z) and np.isnan(score_z)}")
    
    p_with_neg = np.array([10.0, 12.0, -5.0, 15.0] + [20.0]*86)
    slope_n, r2_n, score_n = compute_clenow_momentum(p_with_neg, lookback=90)
    print(f"Price with -5.0: ({slope_n}, {r2_n}, {score_n}) -> All NaN: {np.isnan(slope_n) and np.isnan(r2_n) and np.isnan(score_n)}")
    
    p_with_nan = np.array([10.0, 12.0, np.nan, 15.0] + [20.0]*86)
    slope_nan, r2_nan, score_nan = compute_clenow_momentum(p_with_nan, lookback=90)
    print(f"Price with NaN: ({slope_nan}, {r2_nan}, {score_nan}) -> All NaN: {np.isnan(slope_nan) and np.isnan(r2_nan) and np.isnan(score_nan)}")
    
    # Volatility module edge tests
    vol_zero = calculate_realized_volatility(p_with_zero)
    vol_down_zero = calculate_downside_deviation(p_with_zero)
    print(f"Volatility with 0.0: vol_ann={vol_zero}, vol_down={vol_down_zero} -> Clean NaN: {np.isnan(vol_zero) and np.isnan(vol_down_zero)}")
    
    # Classical momentum edge tests
    mom_zero = compute_jegadeesh_titman_momentum(p_with_zero)
    print(f"Classical Mom with 0.0: {mom_zero} -> Clean NaN: {np.isnan(mom_zero)}")
    
    guards_passed = (
        math.isinf(slope_ovf) and
        slope_udf == -1.0 and
        np.isnan(slope_z) and
        np.isnan(slope_n) and
        np.isnan(slope_nan) and
        np.isnan(vol_zero) and
        np.isnan(mom_zero)
    )
    print(f"Boundary & Overflow Guards Passed: {guards_passed}")
    results["guards_passed"] = guards_passed
    return results


def test_noise_vs_trend_r2_collapse() -> Dict[str, Any]:
    print_section("3. HIGH FREQUENCY NOISE VS PURE TREND (R^2 COLLAPSE)")
    results = {}
    
    np.random.seed(42)
    n_days = 90
    t = np.arange(n_days, dtype=np.float64)
    drift = 0.002  # 0.2% daily drift (~64.8% annualized)
    
    # 3.1 Deterministic Compounding Trend
    p_pure = 100.0 * np.exp(drift * t)
    s_pure, r2_pure, sc_pure = compute_clenow_momentum(p_pure, lookback=90)
    print("3.1 Pure Compounding Trend (Drift=0.2%/day):")
    print(f"  Annualized Slope: {s_pure:.6f} (Theory: {math.exp(250*drift)-1:.6f})")
    print(f"  R^2 Fit:          {r2_pure:.6f} (Theory: 1.000000)")
    print(f"  Clenow Score:     {sc_pure:.6f}")
    
    # 3.2 High Frequency Stationary White Noise Around Flat Price
    white_r2_list, white_score_list = [], []
    for _ in range(500):
        p_white = 100.0 + np.random.normal(0, 2.0, n_days)
        sl, r2, sc = compute_clenow_momentum(p_white, lookback=90)
        white_r2_list.append(r2)
        white_score_list.append(sc)
    avg_white_r2 = float(np.mean(white_r2_list))
    avg_white_sc = float(np.mean(white_score_list))
    print(f"\n3.2 Stationary White Noise (P = 100 + eps, eps ~ N(0, 2), 500 runs):")
    print(f"  Mean R^2:         {avg_white_r2:.6f} (Theory: ~0.01)")
    print(f"  Mean Score:       {avg_white_sc:.6f} (Theory: ~0.00)")
    
    # 3.3 Choppy Sinusoidal Cycle (Zero Net Trend)
    p_sine = 100.0 + 10.0 * np.sin(2 * np.pi * t / 20.0)
    sl_sine, r2_sine, sc_sine = compute_clenow_momentum(p_sine, lookback=90)
    print(f"\n3.3 Choppy Sinusoidal Cycle (20d cycle, zero net drift):")
    print(f"  R^2:              {r2_sine:.6f} (Theory: ~0.00)")
    print(f"  Clenow Score:     {sc_sine:.6f} (Theory: ~0.00)")
    
    # 3.4 Trend with Increasing Stationary Noise Overlay
    noise_levels = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    sweep_results = []
    for noise_std in noise_levels:
        r2_vals, score_vals = [], []
        for _ in range(300):
            p_trend_noisy = 100.0 * np.exp(drift * t) + np.random.normal(0, noise_std, n_days)
            p_trend_noisy = np.clip(p_trend_noisy, 1.0, None)
            sl, r2, sc = compute_clenow_momentum(p_trend_noisy, lookback=90)
            r2_vals.append(r2)
            score_vals.append(sc)
        sweep_results.append({
            "Noise Std": noise_std,
            "Avg R^2": np.mean(r2_vals),
            "Avg Score": np.mean(score_vals),
        })
    df_sweep = pd.DataFrame(sweep_results)
    print("\n3.4 Trend Noise Overlay Sweep (300 runs per level):")
    print(df_sweep.to_string(index=False))
    
    r2_decay = df_sweep["Avg R^2"].tolist()
    is_r2_monotonically_decaying = all(r2_decay[i] >= r2_decay[i+1] for i in range(len(r2_decay)-1))
    print(f"\nR^2 Decay Monotonicity: {is_r2_monotonically_decaying}")
    
    # 3.5 Jump Move vs Smooth Move Comparison
    print("\n3.5 Jump Penalty vs Smooth Compounding Verification:")
    # Asset A: Smooth 20% gain over 90 days
    ret_smooth = np.full(90, np.log(1.20) / 90.0)
    p_smooth = 100.0 * np.exp(np.cumsum(ret_smooth))
    sl_a, r2_a, sc_a = compute_clenow_momentum(p_smooth, lookback=90)
    jr_a, jp_a = calculate_jump_ratio_penalty(p_smooth, lookback=90)
    
    # Asset B: Flat 89 days, 20% gap on day 90
    ret_gap = np.zeros(90)
    ret_gap[-1] = np.log(1.20)
    p_gap = 100.0 * np.exp(np.cumsum(ret_gap))
    sl_b, r2_b, sc_b = compute_clenow_momentum(p_gap, lookback=90)
    jr_b, jp_b = calculate_jump_ratio_penalty(p_gap, lookback=90)
    
    print(f"  Asset A (Smooth +20%): Score = {sc_a:.4f}, Jump Penalty = {jp_a:.4f} (No penalty)")
    print(f"  Asset B (Gap +20%):    Score = {sc_b:.4f}, Jump Penalty = {jp_b:.4f} (Heavily penalized)")
    
    noise_passed = (
        r2_pure > 0.9999 and
        avg_white_r2 < 0.05 and
        abs(avg_white_sc) < 0.01 and
        r2_sine < 0.01 and
        is_r2_monotonically_decaying and
        r2_decay[-1] < 0.15 and
        sc_a > 100.0 * sc_b and
        jp_a == 1.0 and
        jp_b < 0.10
    )
    print(f"\nNoise vs Trend & Jump Penalty Tests Passed: {noise_passed}")
    results["noise_passed"] = noise_passed
    return results


def test_downside_deviation_asymmetry() -> Dict[str, Any]:
    print_section("4. DOWNSIDE SEMI-DEVIATION VS UPSIDE VOLATILITY ASYMMETRY")
    results = {}
    
    np.random.seed(42)
    n_days = 90
    
    # Case 1: Pure Positive Volatility (Monotonically rising with varying positive steps)
    pos_steps = np.random.uniform(0.001, 0.02, n_days)
    p_pure_up = 100.0 * np.exp(np.insert(np.cumsum(pos_steps), 0, 0.0))
    vol_ann_up = calculate_realized_volatility(p_pure_up, lookback=90)
    vol_down_up = calculate_downside_deviation(p_pure_up, lookback=90)
    print("Case 1: Pure Positive Volatility (Only positive daily returns):")
    print(f"  Annualized Total Volatility: {vol_ann_up:.4f} (> 0)")
    print(f"  Downside Semi-Deviation:     {vol_down_up:.4f} (Must be exactly 0.0)")
    
    # Case 2: Pure Negative Volatility (Monotonically falling with varying negative steps)
    neg_steps = np.random.uniform(-0.02, -0.001, n_days)
    p_pure_down = 100.0 * np.exp(np.insert(np.cumsum(neg_steps), 0, 0.0))
    vol_ann_down = calculate_realized_volatility(p_pure_down, lookback=90)
    vol_down_down = calculate_downside_deviation(p_pure_down, lookback=90)
    print("\nCase 2: Pure Negative Volatility (Only negative daily returns):")
    print(f"  Annualized Total Volatility: {vol_ann_down:.4f} (> 0)")
    print(f"  Downside Semi-Deviation:     {vol_down_down:.4f} (> 0)")
    
    # Case 3: Paired Asymmetric Assets
    # Asset A: Alternating +2% and 0% (Positive Volatility, no drawdown)
    ret_a = np.array([0.02 if i % 2 == 0 else 0.00 for i in range(90)])
    p_a = 100.0 * np.exp(np.insert(np.cumsum(ret_a), 0, 0.0))
    vol_ann_a = calculate_realized_volatility(p_a, lookback=90)
    vol_down_a = calculate_downside_deviation(p_a, lookback=90)
    
    # Asset B: Alternating -2% and 0% (Negative Volatility, pure drawdown)
    ret_b = np.array([-0.02 if i % 2 == 0 else 0.00 for i in range(90)])
    p_b = 100.0 * np.exp(np.insert(np.cumsum(ret_b), 0, 0.0))
    vol_ann_b = calculate_realized_volatility(p_b, lookback=90)
    vol_down_b = calculate_downside_deviation(p_b, lookback=90)
    
    print("\nCase 3: Paired Symmetrical Magnitudes:")
    print(f"  Asset A (+2%/0% alternating): Vol_Ann={vol_ann_a:.4f}, Vol_Down={vol_down_a:.4f}")
    print(f"  Asset B (-2%/0% alternating): Vol_Ann={vol_ann_b:.4f}, Vol_Down={vol_down_b:.4f}")
    print(f"  Total Volatility Equality:    {abs(vol_ann_a - vol_ann_b) < 1e-10}")
    print(f"  Downside Volatility Asymmetry: Vol_Down(A) == 0.0 ({vol_down_a == 0.0}), Vol_Down(B) > 0.0 ({vol_down_b > 0.10})")
    
    downside_passed = (
        vol_down_up == 0.0 and
        vol_ann_up > 0.0 and
        vol_down_down > 0.0 and
        vol_down_a == 0.0 and
        vol_down_b > 0.10 and
        abs(vol_ann_a - vol_ann_b) < 1e-10
    )
    print(f"\nDownside Semi-Deviation Asymmetry Test Passed: {downside_passed}")
    results["downside_passed"] = downside_passed
    return results


def test_microstructure_skip_window_12_1() -> Dict[str, Any]:
    print_section("5. MICROSTRUCTURE 1-MONTH SKIP WINDOW (12-1 MOMENTUM)")
    results = {}
    
    # Total bars = 253 (t = 0 to 252)
    # t=0: P_0 = 100.0 (t-252)
    # t=231: P_231 = 200.0 (t-21, skip point)
    # t=252: P_252 = 100.0 (t-0, latest price after 1-month 50% crash)
    
    n_total = 253
    p = np.zeros(n_total, dtype=np.float64)
    # 0 to 231: linear rise 100 to 200
    p[:232] = np.linspace(100.0, 200.0, 232)
    # 232 to 252: crash from 200 to 100
    p[231:] = np.linspace(200.0, 100.0, 22)
    
    mom_12_1 = compute_jegadeesh_titman_momentum(p, lookback_total=252, skip_recent=21)
    mtf = calculate_multi_timeframe_returns(p, skip_recent=21)
    
    print("Scenario A: +100% rally over 11 months, followed by -50% flash crash in month 12:")
    print(f"  P(t-252) [idx 0]:   {p[0]:.2f}")
    print(f"  P(t-21)  [idx 231]: {p[231]:.2f}")
    print(f"  P(t-0)   [idx 252]: {p[252]:.2f}")
    print(f"  Classical Mom 12-1 (Skipping last 21d): {mom_12_1:+.2%} (Expected: +100.00%)")
    print(f"  Raw Unskipped 12-Month Return (ret_12m): {mtf['ret_12m']:+.2%} (Expected: +0.00%)")
    print(f"  Recent 1-Month Return (ret_1m):         {mtf['ret_1m']:+.2%} (Expected: -50.00%)")
    
    # Scenario B: Flat for 11 months, followed by +100% pump in month 12
    p_b = np.zeros(n_total, dtype=np.float64)
    p_b[:232] = 100.0
    p_b[231:] = np.linspace(100.0, 200.0, 22)
    
    mom_12_1_b = compute_jegadeesh_titman_momentum(p_b, lookback_total=252, skip_recent=21)
    mtf_b = calculate_multi_timeframe_returns(p_b, skip_recent=21)
    
    print("\nScenario B: Flat for 11 months, followed by +100% surge in month 12:")
    print(f"  Classical Mom 12-1 (Skipping last 21d): {mom_12_1_b:+.2%} (Expected: +0.00%)")
    print(f"  Raw Unskipped 12-Month Return (ret_12m): {mtf_b['ret_12m']:+.2%} (Expected: +100.00%)")
    print(f"  Recent 1-Month Return (ret_1m):         {mtf_b['ret_1m']:+.2%} (Expected: +100.00%)")
    
    # Exact index verification
    exact_match_a = abs(mom_12_1 - 1.0) < 1e-10
    exact_match_b = abs(mom_12_1_b - 0.0) < 1e-10
    unskipped_match = abs(mtf["ret_12m"] - 0.0) < 1e-10 and abs(mtf["ret_1m"] - (-0.50)) < 1e-10
    
    skip_passed = exact_match_a and exact_match_b and unskipped_match
    print(f"\nMicrostructure Skip Window Test Passed: {skip_passed}")
    results["skip_passed"] = skip_passed
    return results


def test_cross_sectional_scoring_and_ranking_stability() -> Dict[str, Any]:
    print_section("6. CROSS-SECTIONAL STANDARDIZATION & OUTLIER RESILIENCE")
    results = {}
    
    # 6.1 Extreme Outlier (+1000 sigma) Winsorization
    normal_scores = np.random.normal(0.0, 1.0, 100)
    outlier_scores = normal_scores.copy()
    outlier_scores[0] = 1000.0  # extreme outlier
    outlier_scores[1] = -1000.0  # extreme negative outlier
    
    z_scores = winsorized_z_score(outlier_scores, clip_range=(-3.0, 3.0))
    print("Winsorized Z-Scores on Extreme +/- 1000.0 Outliers:")
    print(f"  Max Z-Score: {np.max(z_scores):.4f} (Must be <= +3.0)")
    print(f"  Min Z-Score: {np.min(z_scores):.4f} (Must be >= -3.0)")
    print(f"  Outlier 1 Z: {z_scores[0]:.4f} (Expected: +3.0)")
    print(f"  Outlier 2 Z: {z_scores[1]:.4f} (Expected: -3.0)")
    
    # 6.2 Zero Variance Universe (All identical)
    identical_vals = np.full(50, 42.0)
    z_identical = winsorized_z_score(identical_vals)
    print(f"Identical Universe Z-Score Std: {np.std(z_identical):.4f}, All Zeros: {np.all(z_identical == 0.0)}")
    
    # 6.3 End-to-end Ranking of 500 Synthetic Stocks
    np.random.seed(123)
    stocks = []
    for i in range(500):
        # Generate realistic metrics
        m = MomentumMetrics(
            symbol=f"STOCK_{i+1:03d}",
            close=float(np.random.uniform(50.0, 2000.0)),
            history_bars=252,
            clenow_slope_ann=float(np.random.normal(0.20, 0.40)),
            clenow_r2=float(np.clip(np.random.normal(0.60, 0.25), 0.0, 1.0)),
            clenow_score=0.0,  # will set
            mom_12_1=float(np.random.normal(0.25, 0.35)),
            mom_6_1=float(np.random.normal(0.12, 0.20)),
            volatility_ann=float(np.clip(np.random.normal(0.25, 0.10), 0.05, 1.0)),
            natr_14=float(np.random.uniform(1.5, 5.0)),
            passes_trend=(i % 5 != 0),  # 80% pass trend
        )
        m.clenow_score = m.clenow_slope_ann * m.clenow_r2
        stocks.append(m)
        
    breakdowns, df_ranked = rank_universe(stocks, filter_trend=True)
    print(f"\n500 Stock Universe Ranking:")
    print(f"  Total Stocks:        {len(stocks)}")
    print(f"  Eligible (Trend):    {len(breakdowns)}")
    print(f"  Top 1 Stock:         {breakdowns[0].symbol}, Score: {breakdowns[0].composite_score:.4f}, Rank: {breakdowns[0].rank}, Action: {breakdowns[0].action.value}")
    print(f"  Top 10 Stock:        {breakdowns[9].symbol}, Score: {breakdowns[9].composite_score:.4f}, Rank: {breakdowns[9].rank}, Action: {breakdowns[9].action.value}")
    print(f"  Rank 11 Stock:       {breakdowns[10].symbol}, Score: {breakdowns[10].composite_score:.4f}, Rank: {breakdowns[10].rank}, Action: {breakdowns[10].action.value}")
    print(f"  Last Ranked Stock:   {breakdowns[-1].symbol}, Score: {breakdowns[-1].composite_score:.4f}, Rank: {breakdowns[-1].rank}, Action: {breakdowns[-1].action.value}")
    
    # Check monotonicity of ranks vs composite scores
    scores_ranked = [b.composite_score for b in breakdowns]
    ranks = [b.rank for b in breakdowns]
    is_strictly_sorted = all(scores_ranked[i] >= scores_ranked[i+1] for i in range(len(scores_ranked)-1))
    ranks_correct = (ranks == list(range(1, len(breakdowns)+1)))
    top_10_buys = all(b.action == RebalanceAction.BUY for b in breakdowns[:10])
    beyond_10_holds = all(b.action == RebalanceAction.HOLD for b in breakdowns[10:])
    
    ranking_passed = (
        abs(z_scores[0] - 3.0) < 1e-10 and
        abs(z_scores[1] - (-3.0)) < 1e-10 and
        np.all(z_identical == 0.0) and
        is_strictly_sorted and
        ranks_correct and
        top_10_buys and
        beyond_10_holds
    )
    print(f"\nScoring & Ranking Stability Test Passed: {ranking_passed}")
    results["ranking_passed"] = ranking_passed
    return results


def main():
    print("=" * 80)
    print("  EMPIRICAL BOUNDARY & NUMERICAL STABILITY STRESS HARNESS")
    print("  Milestone 1 Quantitative Momentum Scoring & Ranking Core Engine")
    print("=" * 80)
    
    r1 = test_extreme_price_scale_invariance()
    r2 = test_overflow_and_invalid_boundaries()
    r3 = test_noise_vs_trend_r2_collapse()
    r4 = test_downside_deviation_asymmetry()
    r5 = test_microstructure_skip_window_12_1()
    r6 = test_cross_sectional_scoring_and_ranking_stability()
    
    all_passed = (
        r1.get("scale_invariance_passed", False) and
        r2.get("guards_passed", False) and
        r3.get("noise_passed", False) and
        r4.get("downside_passed", False) and
        r5.get("skip_passed", False) and
        r6.get("ranking_passed", False)
    )
    
    print("\n" + "=" * 80)
    print(f"  FINAL EMPIRICAL STRESS TEST SUITE VERDICT: {'ALL TESTS PASSED (APPROVE)' if all_passed else 'FAILURES DETECTED (REQUEST_CHANGES)'}")
    print("=" * 80)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

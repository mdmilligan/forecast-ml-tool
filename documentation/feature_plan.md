# Feature Implementation Plan for Improved Market Returns

**Goal:** To systematically refine the existing feature set and explore new features to improve the predictive power of the trading model, leading to better signal generation and market returns, specifically considering the 30-minute operational timeframe.

**Guiding Principles:**
1.  **Relevance to Returns:** Prioritize features that have a clear, intuitive, or empirically supported link to price movements and potential trading opportunities on a 30-minute scale.
2.  **Signal Clarity:** Aim for features that can help the model distinguish between noise and genuine trading signals more effectively within intraday movements.
3.  **Iterative Improvement:** Implement and test feature changes one or a few at a time to isolate their impact.
4.  **Balance Complexity and Performance:** While sophisticated features can be powerful, start with simpler, robust features and add complexity if proven beneficial.
5.  **Data-Driven Decisions:** Use feature importance, model performance metrics, and backtest results to guide feature selection and refinement.

---

### I. Analyze & Refine Existing High-Importance Features (30-minute context)

These features are already contributing significantly, but minor refinements might enhance their impact on the 30-minute timeframe.
*   **`spy_volume` (Importance: 2392):**
    *   **Action:** Consider normalizing volume (e.g., as a Z-score over a rolling window of 30-min bars, or volume relative to its own moving average of 30-min bars). This can make volume spikes more comparable across different intraday periods.
*   **Candlestick Features (`upper_wick`, `lower_wick`, `candle_body`, `candle_relative_position` - Importance: ~700-1300):**
    *   **Action:** These are clearly important for 30-minute bars.
        *   **Experiment:** Explore creating explicit features for common 30-minute candlestick patterns that often signal intraday reversals or continuations (e.g., "30-min Engulfing Pattern", "30-min Doji", "30-min Hammer").
*   **`bb_bandwidth` (Importance: 823):**
    *   **Action:** Good for capturing 30-minute volatility contraction/expansion.
    *   **Experiment:** Consider adding a feature for the *rate of change* of `bb_bandwidth` over a few 30-minute periods to detect sudden squeezes or expansions.
*   **`skewness_1d` (Importance: 793 - calculated over 26 of 30-min bars):**
    *   **Action:** Indicates asymmetry in recent returns.
    *   **Experiment:** Ensure this feature is robust. Explore different window lengths for skewness calculation (e.g., `skewness_X_bars` where X is a number of 30-min bars, like 6 or 13 bars for half-day/full-day context if 26 is too long or noisy).
*   **`roc` (Rate of Change - Importance: 701 - current period 10 x 30-min bars):**
    *   **Action:** Fundamental momentum indicator.
    *   **Experiment:** Test different `roc_period` values (e.g., shorter like 3-6 bars for faster intraday signals, longer like 13-26 bars for confirming intraday trends).
*   **`admf` (Accumulation/Distribution Momentum Flow - Importance: 698):**
    *   **Action:** Custom momentum indicator.
    *   **Experiment:** Review its parameters (`admf_weight`, `admf_length`, `admf_price_enable`) in the context of 30-minute data. Small changes here could significantly alter its behavior.
*   **`5D_Slope` (Importance: 643 - calculated over 5 * 13 of 30-min bars) & other MA Slopes:**
    *   **Action:** Good for trend direction.
    *   **Experiment:** Ensure the `lookback` period for slope calculation (currently 5 "days" worth of 30-min bars) is optimal. Test shorter/longer lookbacks in terms of 30-minute bar counts.
*   **`vix_close` (Importance: 639):**
    *   **Action:** Key market sentiment/volatility indicator. Keep. Consider if its raw value or a smoothed version (over a few 30-min bars) is better.
*   **`proximity_bb_1d_upper/lower` (Importance: ~610-630 - BB window 13 of 30-min bars):**
    *   **Action:** Useful for mean-reversion or breakout signals from "daily-equivalent" Bollinger Bands.
    *   **Experiment:** Check if the 13-period BB parameters are optimal for this 30-minute derived signal.
*   **`atr_20`, `volatility` (Importance: ~600 - calculated over 30-min bars):**
    *   **Action:** Standard volatility measures. Keep. Ensure window lengths are appropriate for 30-min noise.
*   **`fisher`, `fisher_trigger` (Importance: ~560-580 - length 10 of 30-min bars):**
    *   **Action:** Oscillator for identifying price reversals.
    *   **Experiment:** The `fisher_length` (currently 10 x 30-min bars) could be tuned for intraday responsiveness.

---

### II. Address Low/Zero Importance Features (30-minute context)

These features might be adding noise or are not configured effectively for 30-minute data.
*   **`bounce_` features (e.g., `bounce_SMA5`, `bounce_strength_SMA100` - Importance: 0 to ~200):**
    *   **Action:** Consistently low. The logic in `detect_bounces` or `threshold_pct` might not be well-calibrated for 30-minute price action.
    *   **Experiment 1 (Simplify/Replace):** Focus on `proximity_` features which seem more effective.
    *   **Experiment 2 (Refine):** If kept, adjust `threshold_pct` for 30-minute volatility or simplify the bounce condition.
    *   **Decision:** Likely prune if simplification/refinement doesn't improve importance.
*   **`market_state` (as a feature - Importance: 25):**
    *   **Action:** The complex logic in `classify_market_state` isn't translating to feature importance on 30-min data.
    *   **Experiment 1 (Simplify):** Try a simpler regime filter based on price relative to a longer-term MA (e.g., 65-period or 130-period MA on 30-min bars, equivalent to 0.5-1 day MA).
    *   **Experiment 2 (Target Variable):** As per the main roadmap, consider predicting this as a target.
*   **`candle_direction` (Importance: 12):**
    *   **Action:** Surprisingly low.
    *   **Investigation:** It's a simple binary; perhaps the more nuanced candle features (body, wicks) capture this information better for 30-min bars. Consider removing if redundant.
*   **Binary Threshold Features (`*_above_zero`, `*_above_50`, `ursi_above_50` - Importance: 1 to 5):**
    *   **Action:** Likely redundant if their continuous counterparts are present and more important.
    *   **Experiment:** Remove these binary versions.
*   **`autocorr_30m` (Importance: 0):**
    *   **Action:** Prune.
*   **Low Importance Statistical Features (`entropy_1d`, `percentile_1d`):**
    *   **Action:** While `skewness_1d` is high, these are low.
    *   **Experiment:** Test removing them.

---

### III. New Feature Ideas & Enhancements (30-minute context)

Explore these to potentially capture new sources of alpha or improve signal timing on 30-minute data.
1.  **Intraday Volatility Dynamics:**
    *   **Feature Idea:** Ratio of short-term volatility (e.g., ATR over 5-10 of 30-min bars) to longer-term intraday volatility (e.g., ATR over 20-40 of 30-min bars).
    *   **Feature Idea:** "Volatility Breakout (30-min)": A binary feature indicating if current 30-min ATR significantly exceeds its recent average (e.g., over last 10-20 bars).
2.  **Intraday Trend Strength & Confirmation:**
    *   **Feature Idea:** ADX (Average Directional Index) with DI+/DI- lines, using periods suitable for 30-min charts (e.g., ADX(7) or ADX(10) instead of the common ADX(14) for daily).
    *   **Feature Idea:** Number of consecutive 30-minute bars price has closed above/below a key short-term MA (e.g., 7-period or 10-period EMA on 30-min data).
3.  **Intraday Relative Strength:**
    *   **Feature Idea:** Price relative to its N-period high/low (e.g., N=10, 20, or 40 of 30-min bars).
4.  **Time-Based Features (Intraday Specific):**
    *   **Feature Idea:** Hour of the day (as categorical or cyclical feature).
    *   **Feature Idea:** Binary features for "first 30-60 mins of market open," "last 30-60 mins before market close," "lunchtime lull" (if applicable).
    *   **Feature Idea:** Time since last N-bar high/low (N being a count of 30-min bars).
5.  **Interaction Features (Carefully Chosen for 30-min data):**
    *   **Feature Idea:** `spy_volume_normalized_30min * roc_30min`.
    *   **Feature Idea:** `fisher_30min * volatility_30min`.
6.  **Rate of Change of Key 30-min Indicators:**
    *   **Feature Idea:** ROC of `ultimate_rsi_30min` or ROC of `admf_30min` (calculated over a few 30-min bars).

---

### IV. Prioritization & Experimentation Strategy for Features (30-minute context)

*   **Tier 1: Quick Wins & Simplification (Highest Priority)**
    1.  Prune confirmed zero/very low importance features.
    2.  Normalize `spy_volume` for intraday comparison.
    3.  Test removing/simplifying `market_state` feature for 30-min data.
    4.  Remove `bounce_` features initially.
*   **Tier 2: Refinement of Existing & High-Potential New Features for 30-min**
    1.  Experiment with 30-minute specific parameters for key existing indicators (`roc_period`, `admf_length`, `fisher_length`, MA slope lookbacks).
    2.  Implement and test explicit 30-minute candlestick pattern features.
    3.  Implement and test ADX with shorter, 30-minute appropriate periods.
    4.  Implement and test intraday volatility dynamics features.
*   **Tier 3: More Exploratory 30-min Features**
    1.  Time-based features (hour of day, market session segments).
    2.  Carefully selected interaction features relevant to intraday dynamics.
    3.  Rate of change of other 30-minute indicators.

**Methodology for Each Feature Change/Addition:**
1.  **Hypothesis:** Why might this change improve returns on a 30-minute timeframe?
2.  **Implement:** Make the code change in `scripts/data_process.py`.
3.  **Retrain Model:** Use the updated feature set.
4.  **Evaluate:**
    *   Check new feature importance.
    *   Compare model metrics (MSE, R2 on validation).
    *   Run backtest and compare key performance indicators.
5.  **Iterate:** Keep, discard, or further refine the feature based on results.
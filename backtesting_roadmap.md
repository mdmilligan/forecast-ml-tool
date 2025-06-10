# Plan: Enhance `scripts/backtest.py` with a Trade Visualization Plot

**1. Goal:**
*   Add a new interactive plot to the existing Dash dashboard in [`scripts/backtest.py`](scripts/backtest.py:1). This plot will display the asset's price chart with clear markers for trade entry and exit points.

**2. Key Features of the New Visualization:**
*   **Base Chart:** The plot will show the historical price series of the traded asset (e.g., `spy_close`).
*   **Trade Markers:**
    *   **Entry Points:** Each trade entry will be marked on the price chart at the specific date and price of entry. We can use a distinct marker, for example, a green upward-pointing triangle (▲).
    *   **Exit Points:** Each trade exit will be marked similarly, perhaps with a red downward-pointing triangle (▼).
*   **Interactivity:** Being part of the Plotly Dash dashboard, this chart will inherently be interactive (zoom, pan, hover-to-see-details).

**3. Data Source for the Plot:**
*   The `Backtester` class in [`scripts/backtest.py`](scripts/backtest.py:8) already calculates and stores trade-specific information in its `trade_stats` dictionary within the [`calculate_returns`](scripts/backtest.py:13) method. This includes:
    *   `trade_stats['entry_dates']`
    *   `trade_stats['entry_prices']`
    *   `trade_stats['exit_dates']`
    *   `trade_stats['exit_prices']`
*   The main price data (e.g., `df['spy_close']`) is also readily available within the backtesting process.

**4. Proposed Implementation Steps:**

*   **A. Modify `calculate_metrics` method (in [`scripts/backtest.py`](scripts/backtest.py:101)):**
    *   Augment the `metrics` dictionary that this method produces. It will be updated to include structured lists of entry and exit points (date and price) suitable for plotting. For example:
        *   `metrics['entry_markers'] = [{'date': d, 'price': p} for d, p in zip(trade_stats['entry_dates'], trade_stats['entry_prices'])]`
        *   `metrics['exit_markers'] = [{'date': d, 'price': p} for d, p in zip(trade_stats['exit_dates'], trade_stats['exit_prices'])]`

*   **B. Modify `create_interactive_dashboard` method (in [`scripts/backtest.py`](scripts/backtest.py:163)):**
    *   A new `plotly.graph_objects.Figure` will be created for this visualization.
    *   It will include:
        1.  A line trace for the asset's price (e.g., `df.index` for x-axis, `df['spy_close']` for y-axis).
        2.  A scatter trace for entry markers using data from `metrics['entry_markers']`.
        3.  A scatter trace for exit markers using data from `metrics['exit_markers']`.
    *   This new graph will be added as a new `dbc.Row(dbc.Col(dcc.Graph(figure=trade_markers_fig)))` to the Dash app's layout, similar to how other plots are added.

**5. Visualizing the Dashboard Structure (Mermaid Diagram):**

```mermaid
graph TD
    A[Backtest Results Dashboard] --> B(Metrics Cards: Returns, Risk, Trades, Performance)
    A --> C[Equity Curve Plot]
    A --> D[Drawdown Plot]
    A --> F[Price Chart with Trade Markers]
    A --> E[Confidence Scores Plot (Optional)]

    F --> F1[Price Line (e.g., spy_close)]
    F --> F2[Entry Markers (Green ▲)]
    F --> F3[Exit Markers (Red ▼)]
```

**6. Benefits of this Approach:**
*   Leverages the existing robust structure of [`scripts/backtest.py`](scripts/backtest.py:1).
*   Integrates seamlessly into the current interactive dashboard.
*   Directly addresses the need for visualizing trade entries/exits.
*   The other "typical valuable backtest data" (metrics like Sharpe, drawdown, win rate, etc.) are already well-covered by this script.
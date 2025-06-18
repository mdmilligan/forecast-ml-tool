# Enhanced Roadmap for Improving Model Training & Backtest Results

```mermaid
graph TD
    A[Current State: Poor Backtest Results] --> B{Problem Diagnosis & Setup};
    B --> BA[Phase 0: Experimentation Framework & Model Management];
    BA --> C1[Phase 1: Feature Engineering & Selection];
    BA --> C2[Phase 2: Target Variable Refinement];
    BA --> C3[Phase 3: Signal Generation & Thresholding];
    BA --> C4[Phase 4: Model Tuning & Evaluation];
    BA --> C5[Phase 5: Advanced Backtesting Analysis];

    C1 --> D1_1[Review High Importance Features];
    C1 --> D1_2[Iteratively Remove/Refine Low Importance Features];
    C1 --> D1_3[Explore Feature Interactions];
    C1 --> D1_4[Consider Feature Scaling/Transformation];

    C2 --> D2_1[Experiment with Predicting Trade Exits];
    C2 --> D2_2[Experiment with Predicting Market Regimes as Primary Target];
    C2 --> D2_3[Refine Definition of 'Return' (e.g., risk-adjusted)];

    C3 --> D3_1[Calibrate Signal Thresholds (Percentile, Fixed)];
    C3 --> D3_2[Refine Confidence Score Logic];
    C3 --> D3_3[Incorporate Feature Values into Signal Logic (e.g., regime-specific signals)];
    C3 --> D3_4[Review Min Hold Period];

    C4 --> D4_1[Expand Hyperparameter Search Space];
    C4 --> D4_2[Refine Custom Scoring Function];
    C4 --> D4_3[Cross-validation Strategy Review (e.g., TimeSeriesSplit)];
    C4 --> D4_4[Consider Simpler Models as Baselines];

    C5 --> D5_1[Detailed Trade-by-Trade Analysis];
    C5 --> D5_2[Analyze Feature Values at Entry/Exit Points];
    C5 --> D5_3[Sensitivity Analysis of Parameters];

    D1_1 & D1_2 & D1_3 & D1_4 --> E{Iterative Retraining & Evaluation Loop};
    D2_1 & D2_2 & D2_3 --> E;
    D3_1 & D3_2 & D3_3 & D3_4 --> E;
    D4_1 & D4_2 & D4_3 & D4_4 --> E;
    E --> F[Improved Backtest Results?];
    F -- Yes --> G[Deploy/Monitor];
    F -- No --> BA;
    C5 --> BA;
```

## Detailed Steps:

### Phase 0: Experimentation Framework & Model Management Strategy
This phase is about setting up a systematic approach to iterating and managing your models.
1.  **Experiment Tracking**:
    *   Enhance `performance_history.csv` to include experiment IDs, Git commit hashes, descriptions of changes, key parameters, all relevant training and backtest metrics, and paths to saved artifacts.
2.  **Version Control**:
    *   Use Git for all code changes. Consider DVC for data/models if needed later.
3.  **Structured Experimentation Workflow**:
    *   Isolate changes (one major variable per experiment).
    *   Always compare against baseline and previous best.
    *   Formulate a clear hypothesis for each experiment.
4.  **Model Naming and Storage**:
    *   Implement a consistent naming convention for saved model artifacts (e.g., including experiment ID).
    *   Organize saved models systematically.
5.  **Dedicated Hold-Out/Forward-Testing Set**:
    *   Maintain the concept of splitting data into Training, Validation, and a Final Hold-Out Test set to ensure robust evaluation. Your current 90/10 split can serve as training/validation for iterative development, reserving a later, untouched data segment for the final model's true out-of-sample test.

### Phase 1: Feature Engineering & Selection
*   **Review High Importance Features**: Understand why dominant features are selected.
*   **Iteratively Remove/Refine Low Importance Features**: Prune features with zero or very low importance. Re-evaluate binary indicators and consider if raw values are more informative. Review parameters of technical indicators.
*   **Explore Feature Interactions**: Explicitly create interaction terms for high-importance features.
*   **Consider Feature Scaling/Transformation**: Explore alternatives to `StandardScaler` for specific features (e.g., `QuantileTransformer`, `PowerTransformer`).

### Phase 2: Target Variable Refinement
*   **Experiment with Predicting Trade Exits**: Utilize existing `y_exit_long`, `y_exit_short` for multi-target learning or direct classification.
*   **Experiment with Predicting Market Regimes as Primary Target**: If `market_state` is conceptually sound, try predicting it as a target to inform regime-specific models or logic.
*   **Refine Definition of 'Return'**: Consider predicting risk-adjusted returns or returns conditional on volatility.

### Phase 3: Signal Generation & Thresholding
*   **Calibrate Signal Thresholds**: Experiment with different percentile values for return predictions, test fixed thresholds, and explore adaptive thresholds based on volatility or market regime.
*   **Refine Confidence Score Logic**: Consider alternatives like prediction variance (for ensembles), distance from decision boundary (for classification), or integrating feature-based confidence.
*   **Incorporate Feature Values into Signal Logic**: Develop different signal logic or thresholds for different market regimes if they can be reliably classified.
*   **Review Min Hold Period**: Analyze the impact of `min_hold_bars` on trade quality and overall performance.

### Phase 4: Model Tuning & Evaluation
*   **Expand Hyperparameter Search Space**: Consider wider ranges or more iterations for `RandomizedSearchCV`.
*   **Refine Custom Scoring Function**: Analyze the sensitivity of `position_aware_score` to its penalty/reward factors.
*   **Cross-validation Strategy Review**: Implement `TimeSeriesSplit` from scikit-learn for more robust time-series cross-validation.
*   **Consider Simpler Models as Baselines**: Test simpler models (e.g., Logistic Regression, Linear Regression) with a reduced feature set as a performance floor and sanity check.

### Phase 5: Advanced Backtesting Analysis
*   **Detailed Trade-by-Trade Analysis**: For losing or missed trades, examine feature values and model predictions using `data/test_features.csv` and `data/test_predictions.csv`.
*   **Analyze Feature Values at Entry/Exit Points**: Plot distributions of key feature values when signals are generated.
*   **Sensitivity Analysis of Parameters**: Assess how results change with small variations in signal thresholds, confidence thresholds, and `min_hold_bars`.

### Iterative Retraining & Evaluation Loop:
This is the core of the process. After each change in Phases 1-4:
1.  Retrain the model using the strategy outlined in Phase 0.
2.  Generate new predictions.
3.  Run the backtest.
4.  Log all results meticulously.
5.  Compare to the baseline and previous experiments.
6.  Decide on the next step based on the outcome and your hypotheses.

### Retraining Strategy (for a "Deployed" Model - Future Consideration):
This includes monitoring performance, detecting concept drift, defining retraining triggers (scheduled, performance-based, event-based), and the process for retraining (full pipeline, comparison against previous version).
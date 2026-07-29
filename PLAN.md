# PatchTST Validation Plan

## 1. Objective

Validate whether PatchTST can accurately forecast the future return distribution of a market asset using only information available at prediction time.

The primary comparison is:

```text
PatchTST predicted return
versus
actual realized future return
```

No competing machine-learning model is required for the initial validation.

---

## 2. Primary prediction target

For a four-hour horizon:

\[
y_{t,4} = \log\left(\frac{C_{t+4}}{C_t}\right)
\]

Where:

* \(C_t\) is the close at prediction time.
* \(C_{t+4}\) is the close four candles later.
* Each candle represents one hour.

Example:

```text
Current close: $100
Close after four hours: $101

Actual log return:
log(101 / 100) ≈ 0.00995
```

The PatchTST prediction is compared directly with `0.00995`.

---

## 3. Final desired output

```json
{
  "horizon": "4h",
  "expected_log_return": 0.006,
  "probability_above_cost_threshold": 0.67,
  "return_p10": -0.008,
  "return_p50": 0.004,
  "return_p90": 0.019,
  "expected_volatility": 0.012,
  "expected_max_adverse_excursion": -0.009,
  "expected_max_favorable_excursion": 0.017,
  "regime_probabilities": {
    "uptrend": 0.55,
    "downtrend": 0.15,
    "sideways": 0.30
  }
}
```

Validation must compare every predicted field with a corresponding realized value.

---

# 4. Explicit validation reference

## Ground truth

The main validation reference is the actual market outcome following each prediction timestamp.

For every timestamp, save:

```text
Prediction timestamp
Predicted expected return
Predicted return quantiles
Predicted probability above cost
Predicted volatility
Predicted MAE
Predicted MFE
Predicted regime probabilities

Actual four-hour return
Actual four-hour volatility
Actual MAE
Actual MFE
Actual regime
```

Example validation row:

```json
{
  "timestamp": "2026-01-15T10:00:00Z",
  "predicted_log_return": 0.006,
  "actual_log_return": 0.0048,
  "predicted_p10": -0.008,
  "predicted_p50": 0.004,
  "predicted_p90": 0.019,
  "actual_above_cost": true,
  "actual_volatility": 0.010,
  "actual_mae": -0.006,
  "actual_mfe": 0.013,
  "actual_regime": "uptrend"
}
```

## Optional sanity reference

A zero-return prediction can be reported as a sanity reference:

```text
predicted return = 0
```

This is not a competing model. It only shows whether PatchTST predicts returns better than always assuming no price change.

The principal result remains PatchTST predictions versus actual returns.

---

# 4a. Competing baselines

The zero predictor is a floor, not a competitor. Beating it does not show that a
transformer is the right model, only that the predictions are not worse than
nothing. Report these competing models alongside PatchTST at every stage:

```text
momentum_raw    trailing four-hour log return, unscaled (pure persistence)
momentum_ols    the same signal with a train-fitted intercept and slope
ridge_last      ridge on the feature vector at time t only, no history
ridge_lags      ridge on the features at ten log-spaced lags across the window
gbm_last        gradient boosting on the feature vector at time t only
gbm_lags        gradient boosting on the same ten log-spaced lags
```

Every baseline must see exactly the same folds, the same purged boundaries, the
same feature set and the same train-only normalization as PatchTST, and be scored
by the same metrics in raw log-return units. Baselines are deterministic, so they
need no seed averaging; PatchTST is compared using its seed-averaged predictions.

The `_last` variants carry no history at all. They exist to answer the question the
zero predictor cannot: how much of the measured skill needs a sequence model, and
how much is available from the most recent candle.

## Incremental value

Correlation with a baseline is not enough on its own. Also report, aligned on
`(symbol, timestamp)`:

```python
pred_corr     = corr(patchtst_pred, baseline_pred)
residual      = actual - ols_fit(baseline_pred -> actual)
resid_pearson = corr(patchtst_pred, residual)
combo_pearson = corr(zscore(patchtst_pred) + zscore(baseline_pred), actual)
```

`resid_pearson` is the honest measure of what PatchTST adds. If it is near zero the
transformer is re-deriving the baseline. If `combo_pearson` beats both models alone,
the two carry different information and the transformer is worth keeping even when
it loses on raw correlation.

## Shuffled-target control

Independently of the baselines, refit one fold with the target randomly permuted
within each symbol. This establishes the null distribution the pipeline produces on
noise. Any reported correlation must be interpreted against it.

---

# 5. Input features

Use normalized, causal features calculated only from completed candles.

## Feature group A: candle price structure

Mandatory features:

```text
close_return
open_gap
high_return
low_return
candle_body
candle_range
close_location
upper_wick
lower_wick
```

Definitions:

```python
close_return = log(close_t / close_t_minus_1)

open_gap = log(open_t / close_t_minus_1)

high_return = log(high_t / close_t_minus_1)

low_return = log(low_t / close_t_minus_1)

candle_body = log(close_t / open_t)

candle_range = log(high_t / low_t)

close_location = (
    close_t - low_t
) / max(high_t - low_t, epsilon)

upper_wick = (
    high_t - max(open_t, close_t)
) / close_t_minus_1

lower_wick = (
    min(open_t, close_t) - low_t
) / close_t_minus_1
```

Avoid using raw prices as the primary representation when training across different assets.

---

## Feature group B: momentum

Include cumulative returns over several historical windows:

```text
return_2
return_4
return_8
return_12
return_24
return_48
return_168
```

Example:

```python
return_24 = log(close_t / close_t_minus_24)
```

These describe short-term, daily and weekly price movement.

---

## Feature group C: volume

Include:

```text
log_volume
volume_change
volume_zscore_24
volume_zscore_72
volume_zscore_168
dollar_volume
relative_volume
```

Definitions:

```python
log_volume = log1p(volume_t)

volume_change = log(
    (volume_t + epsilon) /
    (volume_t_minus_1 + epsilon)
)

dollar_volume = log1p(close_t * volume_t)
```

Volume z-scores must use trailing windows only.

---

## Feature group D: volatility

Include:

```text
realized_volatility_4
realized_volatility_12
realized_volatility_24
realized_volatility_48
realized_volatility_168
average_range_12
average_range_24
average_range_168
```

Example:

```python
realized_volatility_24 = sqrt(
    sum(last_24_log_returns ** 2)
)
```

Also include:

```python
normalized_range = (high_t - low_t) / close_t
```

---

## Feature group E: price-volume interaction

Include:

```text
return_x_volume_zscore
range_x_volume_zscore
signed_volume_change
```

Definitions:

```python
return_x_volume_zscore = (
    close_return * volume_zscore_24
)

range_x_volume_zscore = (
    normalized_range * volume_zscore_24
)

signed_volume_change = (
    sign(close_return) * abs(volume_change)
)
```

These explicitly expose relationships between price movement and activity.

---

## Feature group F: benchmark context

For stocks:

```text
market_index_return
sector_etf_return
asset_minus_market_return
market_realized_volatility
```

For cryptocurrency:

```text
btc_return
eth_return
asset_minus_btc_return
btc_realized_volatility
```

Example:

```python
relative_return = asset_return - benchmark_return
```

Benchmark data must be timestamp-aligned and available at prediction time.

---

## Feature group G: calendar context

For hourly candles:

```text
hour_sin
hour_cos
day_of_week_sin
day_of_week_cos
```

For stocks:

```text
minutes_from_open
minutes_to_close
opening_hour
closing_hour
```

Use cyclical encoding:

```python
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)
```

---

# 6. Feature validation stages

Validate features progressively.

| Version | Features                               |
| ------- | -------------------------------------- |
| F0      | Close returns only                     |
| F1      | OHLC candle structure and volume       |
| F2      | F1 plus momentum and volatility        |
| F3      | F2 plus benchmark context              |
| F4      | F3 plus calendar features              |
| F5      | F4 plus order-book or derivatives data |

Use **F2 as the primary initial feature set**.

Do not introduce order books, funding rates or news until the OHLCV model has been validated.

---

# 7. Dataset format

For each timestamp \(t\):

```text
Input:
Previous 256 completed hourly candles

Targets:
Return during the next four hours
Volatility during the next four hours
MAE during the next four hours
MFE during the next four hours
Regime during the next four hours
```

Example input shape:

```text
[batch_size, 256, feature_count]
```

For 24 features and batch size 16:

```text
[16, 256, 24]
```

---

# 8. Target construction

## Actual four-hour return

```python
actual_return = log(close_t_plus_4 / close_t)
```

## Actual future volatility

```python
future_returns = [
    log(close_t_plus_1 / close_t),
    log(close_t_plus_2 / close_t_plus_1),
    log(close_t_plus_3 / close_t_plus_2),
    log(close_t_plus_4 / close_t_plus_3),
]

actual_volatility = sqrt(
    sum(r ** 2 for r in future_returns)
)
```

## Actual maximum adverse excursion

For a long-position interpretation:

```python
actual_mae = min(
    log(low_t_plus_i / close_t)
    for i in range(1, 5)
)
```

## Actual maximum favorable excursion

```python
actual_mfe = max(
    log(high_t_plus_i / close_t)
    for i in range(1, 5)
)
```

## Actual above-cost result

```python
actual_above_cost = (
    actual_return > cost_threshold
)
```

---

# 9. Regime target

Calculate path efficiency:

```python
path_efficiency = (
    abs(actual_return) /
    (
        sum(abs(r) for r in future_returns)
        + epsilon
    )
)
```

Suggested labels:

```text
Uptrend:
actual return > trend threshold
AND path efficiency >= 0.35

Downtrend:
actual return < -trend threshold
AND path efficiency >= 0.35

Sideways:
everything else
```

The threshold should account for volatility and trading costs:

```python
trend_threshold = max(
    cost_threshold,
    0.5 * trailing_volatility * sqrt(4),
)
```

Freeze this definition before evaluating the model.

---

# 10. Time-based splitting

Never randomly shuffle market samples.

Use:

```text
Training period
Validation period
Test period
Final untouched holdout
```

Recommended structure:

```text
Development data: oldest 80–85%
Final holdout: newest 15–20%
```

Within development data, use at least five walk-forward folds.

Example:

```text
Fold 1:
Train → Validation → Test

Fold 2:
Expand training period
Train → Validation → Test

Fold 3:
Expand training period
Train → Validation → Test
```

For hourly data:

```text
Validation window: 3 months
Test window: 3 months
```

No training target may cross into the validation period.

For a four-hour target:

```text
last_training_sample_timestamp
<= training_end - 4 hours
```

---

# 11. Normalization

For each fold:

1. Fit clipping limits using training data only.
2. Fit scalers using training data only.
3. Apply the frozen scaler to validation and test.
4. Never fit normalization using future observations.
5. Convert predictions back to raw log-return units before evaluation.

Suggested transformations:

```text
Returns:
robust scaling

Volume:
log1p followed by robust scaling

Volatility:
log(volatility + epsilon)

MAE and MFE:
robust scaling or raw return units
```

Test PatchTST with RevIN enabled and disabled.

---

# 12. Stage one: expected-return validation

Begin with only:

```json
{
  "expected_log_return": 0.006
}
```

Recommended model configuration for 8 GB VRAM:

```text
Context length: 256
Patch length: 16
Patch stride: 8
Encoder layers: 3
Embedding dimension: 128
Attention heads: 4
Feed-forward dimension: 512
Dropout: 0.10
Batch size: 16
Precision: FP16
```

Use gradient accumulation if needed.

## Loss

Start with Huber loss:

```python
loss = huber_loss(
    predicted_return,
    actual_return,
)
```

Also test MSE as a controlled experiment.

---

# 13. Expected-return evaluation

Compare every prediction directly with its realized return.

Report:

```text
MAE
RMSE
Huber loss
Pearson correlation
Spearman correlation
Directional accuracy
Prediction bias
Calibration slope
```

## Directional accuracy

```python
direction_correct = (
    sign(predicted_return)
    == sign(actual_return)
)
```

Also report direction accuracy only when:

```text
absolute predicted return > trading cost
```

This prevents tiny economically meaningless predictions from dominating the metric.

## Prediction bias

```python
prediction_bias = mean(
    predicted_return - actual_return
)
```

## Calibration slope

Fit on validation predictions:

```text
actual_return =
intercept + slope × predicted_return
```

A well-scaled model should have:

```text
intercept near 0
slope near 1
```

---

# 14. Stage two: return-distribution validation

Once expected-return prediction is stable, use a Student-t output head.

The model predicts:

```text
location
scale
degrees of freedom
```

Derive:

```text
expected_log_return
probability_above_cost_threshold
p10
p50
p90
```

Use negative log likelihood as the training loss.

---

# 15. Quantile validation against actual returns

For each prediction, check the actual realized return.

## p10 coverage

```python
p10_coverage = mean(
    actual_return <= predicted_p10
)
```

Target:

```text
approximately 10%
```

## p50 coverage

```python
p50_coverage = mean(
    actual_return <= predicted_p50
)
```

Target:

```text
approximately 50%
```

## p90 coverage

```python
p90_coverage = mean(
    actual_return <= predicted_p90
)
```

Target:

```text
approximately 90%
```

## Central interval coverage

```python
inside_interval = (
    predicted_p10
    <= actual_return
    <= predicted_p90
)
```

Target:

```text
approximately 80%
```

Report:

```text
Negative log likelihood
CRPS
p10 pinball loss
p50 pinball loss
p90 pinball loss
Coverage
Average interval width
```

A wide interval can achieve good coverage without being useful, so coverage and interval width must be evaluated together.

---

# 16. Probability validation

For each prediction:

```python
actual_event = int(
    actual_return > cost_threshold
)
```

Compare that binary outcome with:

```python
predicted_probability_above_cost
```

Report:

```text
Brier score
Log loss
Calibration curve
Expected calibration error
```

Group predictions into probability buckets:

```text
0.50–0.55
0.55–0.60
0.60–0.65
0.65–0.70
0.70–0.80
0.80–1.00
```

For predictions near `0.70`, approximately 70% should actually exceed the cost threshold.

---

# 17. Volatility validation

Compare:

```text
predicted future volatility
versus
actual future volatility
```

Report:

```text
MAE
RMSE
RMSE of log volatility
QLIKE
Pearson correlation
Spearman correlation
```

Also group results by actual volatility:

```text
Low volatility
Medium volatility
High volatility
Extreme volatility
```

---

# 18. MAE and MFE validation

Compare predicted excursions against actual future highs and lows.

Report:

```text
MAE prediction error
MFE prediction error
Quantile pinball loss
Quantile coverage
Error by volatility regime
```

For example:

```python
mae_error = (
    predicted_expected_mae - actual_mae
)

mfe_error = (
    predicted_expected_mfe - actual_mfe
)
```

Pay special attention to dangerous errors:

```text
Predicted downside is small
but actual downside is large
```

These underpredictions matter more for risk control than conservative overpredictions.

---

# 19. Regime validation

Compare the predicted probabilities against the actual regime label.

Report:

```text
Multiclass log loss
Multiclass Brier score
Macro F1
Balanced accuracy
Per-class precision
Per-class recall
Confusion matrix
```

Also evaluate probability calibration separately for:

```text
Uptrend
Downtrend
Sideways
```

The predicted class is:

```python
predicted_regime = argmax(
    regime_probabilities
)
```

But probability calibration is more important than raw classification accuracy.

---

# 20. Multi-task model

After validating each target separately:

```text
Shared PatchTST encoder
├── Student-t return head
├── Volatility head
├── MAE head
├── MFE head
└── Regime head
```

Initial loss:

\[
L = 1.0 L_{return} + 0.25 L_{volatility} + 0.15 L_{MAE} + 0.15 L_{MFE} + 0.20 L_{regime}
\]

Compare the full model with the earlier return-only PatchTST.

Keep auxiliary tasks only when they do not materially reduce expected-return or return-distribution accuracy.

---

# 21. Ablation experiments

Run controlled tests for:

## Context length

```text
128
256
512
```

## Patch size

```text
Patch 8, stride 4
Patch 16, stride 8
Patch 32, stride 16
```

## Features

```text
F0
F1
F2
F3
F4
```

## Channel interaction

```text
Channel attention disabled
Channel attention enabled
```

## Normalization

```text
RevIN disabled
RevIN enabled
```

## Loss

```text
MSE
Huber
Student-t negative log likelihood
```

Change only one experimental category at a time.

---

# 22. Trading validation

Prediction accuracy and trading usefulness must be reported separately.

Freeze one trading rule before opening the test set.

Example long condition:

```text
Expected log return > cost threshold

Probability above cost >= 0.60

Predicted p10 is above the maximum acceptable downside
```

Example short condition:

```text
Expected log return < negative cost threshold

Probability below negative cost >= 0.60

Predicted p90 is below the maximum acceptable short-side risk
```

For the initial strategy:

```text
Enter at the next candle open
Hold exactly four candles
Use fixed position size
Permit no overlapping position in the same asset
Include fees, spread and slippage
```

Report:

```text
Net return
Average net return per trade
Sharpe ratio
Sortino ratio
Maximum drawdown
Win rate
Profit factor
Number of trades
Turnover
Exposure
Long performance
Short performance
Performance by year
Performance by asset
Performance by volatility regime
```

---

# 23. Acceptance criteria

## Expected return

Accept only if:

* Prediction error is stable across walk-forward periods.
* Pearson and Spearman correlations are positive.
* The prediction is not consistently biased.
* Larger predicted returns correspond to larger realized returns.
* Results are stable across at least five random seeds.
* Performance is not produced by one asset or one short period.
* Correlation exceeds what a shuffled-target run of the same pipeline produces.

Accept PatchTST specifically, rather than the task, only if:

* It beats the section 4a baselines, or adds information they miss
  (`resid_pearson` clearly positive and `combo_pearson` above both models alone).

## Return distribution

Accept only if:

* p10 coverage is reasonably close to 10%.
* p50 coverage is reasonably close to 50%.
* p90 coverage is reasonably close to 90%.
* Central 80% coverage is approximately 75–85%.
* Prediction intervals are not excessively wide.
* Probability predictions have acceptable Brier score and calibration.

## Economic validation

Accept only if:

* Performance remains positive after realistic costs.
* Cost-adjusted performance beats the section 4a baselines on the same windows.
* Results survive multiple market regimes.
* Results are not dependent on a few extreme trades.
* Final holdout performance is consistent with walk-forward results.

---

# 24. Recommended first experiment

```text
Input:
256 hourly candles

Features:
F2 — candle structure, volume, momentum and volatility

Target:
Actual future four-hour log return

Model:
PatchTST

Loss:
Huber

Configuration:
3 encoder layers
128 embedding dimension
4 attention heads
Patch length 16
Stride 8
FP16
Batch size 16

Validation:
Five expanding walk-forward folds

Comparison:
PatchTST predictions directly against actual realized returns

Final holdout:
Newest 15–20% of the timeline
```

The initial question is:

> When PatchTST predicts a four-hour log return, how closely does that prediction correspond to the actual realized four-hour return on completely unseen timestamps?

Only after this is validated should probability, quantile, volatility, excursion and regime outputs be added.

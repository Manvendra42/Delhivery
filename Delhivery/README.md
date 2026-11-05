# Delhivery Last-Mile Delay Intelligence Pipeline

**Capstone project · Synthetic data · Reproducible Python pipeline**

This repository builds an end-to-end shipment delay workflow: clean a messy shipment export, estimate delivery hours, flag likely promise misses, group hubs into operating segments, and turn undelivered shipments into an action list. The data are synthetic; no real Delhivery shipment records are used.

## How to run

Run these commands from the repository root in the listed order. The three generated data files are committed, but the generator can recreate them deterministically.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`package_model.py` regenerates `models/delay_model.pkl` from the pinned Part C configuration and verifies the reloaded model against the in-memory model. A committed pickle is reliably loadable only with the scikit-learn version pinned in `requirements.txt`; if installed versions differ, rerun `python package_model.py` rather than relying on the committed pickle. All CSVs are written without an index. Result decimals are rounded to four places, except the reload check, which retains extra precision to verify equivalence.

## Data findings

- **Cleaning:** 1,224 raw rows contained 24 exact duplicate rows, leaving 1,200 unique shipments. Five scanner-fault weight sentinels were treated as missing. Median fill values were distance 314.95 km, weight 1.83 kg, and declared value INR 1,121.50. The final feature/target null count was 0.
- **Service type:** Express had 119 delayed shipments out of 243 (delay rate 0.4897), the highest of the three service types. Economy was 27/315 (0.0857), and Standard was 132/642 (0.2056).
- **Hub tier:** delay rate rose from 0.0400 for Tier 1 to 0.4400 for Tier 3.
- **Distance:** delay rate was 0.0219 in the 0–200 km band (6/274), 0.1368 in 201–400 km (77/563), 0.4364 in 401–600 km (120/275), and 0.8523 in 601+ km (75/88). Delay rate rose sharply with distance in these synthetic data.
- **Correlations with delivery hours:** the three strongest absolute correlations, excluding `delivery_hours` itself, were `promised_hours` = 0.8372, `distance_km` = 0.4013, and `n_stops` = 0.3247. Correlation is descriptive and does not establish causation.
- **Declared value:** the median-filled `declared_value_inr` mean was INR 1,533.05 and median INR 1,121.50; after natural-log transformation, the mean was 7.0277 and median 7.0224. The log compresses the long right tail so very high declared values exert less influence on the scale.

## Model results (regression)

- The mean-prediction baseline had validation R² = -0.0212; its negative R² means it performed worse than predicting the validation target mean.
- **Linear regression** had the lowest validation RMSE at 4.8912 hours and validation R² = 0.9313. Its one-time final test result was RMSE 4.5235 hours and R² = 0.9493.
- Tree-depth diagnostics show high bias at depth 1 (train RMSE 11.6334, validation RMSE 12.1324) and increasing variance at unlimited depth (train RMSE 0.0000, validation RMSE 7.8368). Depth 6 gave the best validation tree RMSE in this run (6.2796).
- Gradient boosting learning-rate experiment: learning rate 0.1 had the highest validation R² (0.9306). The 0.01 learning rate underfit within 100 estimators (R² 0.7127); 1.0 overshot and performed worse (R² 0.8429).
- Five-fold cross-validation on the combined training-plus-validation data gave linear regression mean R² 0.9261 (std 0.0048) and gradient boosting mean R² 0.9353 (std 0.0045). CV slightly favoured Gradient Boosting, but linear regression won on the held-out validation set and remained the selected model. This illustrates why the model-selection procedure should be decided before looking at the test set.

## Model results (classification)

- The all-on-time baseline achieved accuracy 0.7667 but recall and F1 were both 0.0000: it misses every delayed shipment, so accuracy alone is misleading.
- The five bookkeeping/noise columns together accounted for 0.2245 of the full forest feature importance. All five ranked outside the top seven by F-score; the top seven were `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag`, and `sale_event_flag`.
- SelectKBest validation F1 results were k=5: 0.6988, k=7: 0.8333, k=10: 0.7857, k=12: 0.8049, and k=17: 0.7848. The seven-feature model performed best in this sweep.
- Tuning did not beat the untuned forest on this validation set: grid search F1 was 0.7750, randomized search F1 was 0.7848, and the default seven-feature forest F1 was 0.8333. The validation set contains only 180 parcels, so small changes in a few classifications can change the ranking; tuning is not a guarantee of future improvement.
- The selected alert threshold was 0.5, the highest threshold with validation recall at least 0.80. At that threshold, validation precision was 0.7955 and recall was 0.8333; increasing the threshold improves precision but eventually misses too many delayed parcels.
- On the untouched test set, precision was 0.6889, recall 0.7561, and F1 0.7209. That is about 6.9 truly delayed parcels per 10 flagged, and about 7.6 of every 10 delayed parcels caught. Test recall differs from validation recall; the test set contains only 41 delayed parcels, so each changed outcome has a noticeable effect.

## Operations recommendation

Use the three hub playbooks rather than relying on tier labels alone. The **Escalate** segment has 9 hubs, average delivery time 64.65 hours, and delay rate 0.4600. Its hubs and observed delay rates are:

- HUB13 Nagpur: 0.4600
- HUB15 Kochi: 0.3600
- HUB17 Patna: 0.4600
- HUB18 Ranchi: 0.3800
- HUB19 Guwahati: 0.6200
- HUB20 Raipur: 0.4800
- HUB21 Bhubaneswar: 0.4000
- HUB23 Siliguri: 0.4400
- HUB24 Agartala: 0.5400

Tier/segment mismatches to review:

- **HUB07 Kolkata** is Tier 1 but behaves like **Watch**; the tier-based expectation would have been Standard. Treat its tier label as capacity context, not as a substitute for observed operating performance.
- **HUB13 Nagpur** is Tier 2 but behaves like **Escalate**; the tier-based expectation would have been Watch.
- **HUB15 Kochi** is Tier 2 but behaves like **Escalate**; the tier-based expectation would have been Watch.
- **HUB22 Dehradun** is Tier 3 but behaves like **Watch**; the tier-based expectation would have been Escalate.

The alert threshold is 0.5. On test data, precision 0.6889 means about 6.9 of every 10 flagged parcels are truly delayed; recall 0.7561 means about 7.6 of every 10 delayed parcels are caught. In the 48-parcel new batch, 8 were flagged; 6 need a hub-manager escalation and 2 need a proactive customer call. The remaining 40 receive no action.

K=2 had an across/within separation ratio of 2.894, K=3 was 5.393, K=4 was 6.305, and K=5 was 6.887. Although separation rises for K=4 and K=5, those solutions create clusters with fewer than six hubs, too small to support a stable separate playbook. K=3 also matches the business requirement of three operating playbooks. Segment profiles were: Standard—7 hubs, 47.24 mean hours, 0.0257 delay rate; Watch—8 hubs, 56.81 mean hours, 0.1550 delay rate; Escalate—9 hubs, 64.65 mean hours, 0.4600 delay rate.

## Production plan

- **Accuracy:** regression metrics and classifier precision/recall/F1 are recorded in `outputs/part_b_metrics.csv` and `outputs/part_c_final_test.csv`.
- **Reliability:** `package_model.py` reloads the serialized model and checks probability agreement on the same 180 test rows; the recorded maximum absolute difference is 0.0.
- **Reproducibility:** the generator uses fixed seeds, train/validation/test split seeds are fixed at 42, and every script runs in a documented order.
- **Performance:** inference uses a small tabular feature matrix and a 100-tree random forest; benchmark runtime on the deployment host before setting a production SLA.
- **Maintainability:** shared preprocessing lives in `src/shipment_prep.py`; scoring calls the same `clean_shipments` and `build_features` functions rather than maintaining a second implementation.
- **Consistency:** `models/feature_order.csv` pins the 17 input features, and `models/impute_values.csv` carries historical medians into inference.
- **Monitorability:** output actions include a probability, threshold decision, hub segment, and action, allowing operations to audit each decision.

### Monitoring triggers

- Weekly delayed share among delivered parcels: investigate if it rises above 0.30 or changes by more than 5 percentage points versus the trailing 8-week baseline.
- Weekly share of parcels flagged: investigate if it is outside 10%–40% for two consecutive weeks; check input drift and threshold calibration before changing the threshold.
- Recall on matured parcels (once true outcomes are known): review/retrain if rolling four-week recall falls below 0.75.
- Feature drift: investigate if a key numeric feature’s population stability index exceeds 0.20, or if missingness in distance, weight, or declared value exceeds 5%.

Retrain quarterly, and sooner when a monitoring trigger fires or operations changes materially. For the 15–28 September sale window, compare sale-period outcomes with prior comparable sale events (or matched sale days), not ordinary weeks; preserve `sale_event_flag` and review its interaction with service type. Data drift would appear as changes in distance, stops, service mix, missingness, hub behaviour, or sale-event effects compared with the training distribution. Recalibrate the alert threshold only on a new, temporally separated validation set and keep a final test period untouched.

## Limitations

- The data are synthetic and encode a simplified delivery-time formula; performance may be much lower on real-world operations data.
- The test set contains only 180 parcels and 41 delayed outcomes, so precision and recall have sampling uncertainty.
- The alert threshold is selected from one validation split; a time-based evaluation and repeated backtesting would give a more reliable estimate.
- Hub segments are based on historical averages and K-Means; averages can hide within-hub variation and a hub’s operating conditions can change.
- The model’s bookkeeping columns have non-zero forest importance despite weak univariate F-scores, so they should be excluded or investigated in a production model rather than assumed harmless.

from pathlib import Path
import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, PROMISED_HOURS, clean_shipments, build_features

ROOT = Path(__file__).resolve().parent
for d in ['outputs', 'data']:
    (ROOT / d).mkdir(exist_ok=True)
raw = pd.read_csv(ROOT/'data/raw_shipments.csv')
hubs = pd.read_csv(ROOT/'data/hubs.csv')
dedup = raw.drop_duplicates().copy()
spellings_before = int(dedup['service_type'].nunique())
missing_distance = int(dedup['distance_km'].isna().sum())
missing_weight = int(dedup['parcel_weight_kg'].isna().sum())
missing_value = int(dedup['declared_value'].isna().sum())
missing_payment = int(dedup['payment_mode'].isna().sum())
sentinel_rows = int((pd.to_numeric(dedup['parcel_weight_kg'], errors='coerce') > 100).sum())
clean, medians = clean_shipments(raw)
spellings_after = int(clean['service_type'].nunique())
model = build_features(clean, hubs)
model['is_delayed'] = (model['delivery_hours'] > model['promised_hours']).astype(int)
model.to_csv(ROOT/'data/modelling_table.csv', index=False, float_format='%.4f')
logv = np.log(model['declared_value_inr'].clip(lower=1))
q1, q3 = model['delivery_hours'].quantile([.25,.75])
iqr = q3-q1
checks = {
 'raw_rows':len(raw), 'duplicate_rows_removed':len(raw)-len(dedup), 'rows_after_dedup':len(dedup),
 'missing_distance_km_before':missing_distance, 'missing_parcel_weight_before':missing_weight,
 'missing_declared_value_before':missing_value, 'missing_payment_mode_before':missing_payment,
 'sentinel_weight_rows':sentinel_rows, 'service_type_spellings_before':spellings_before,
 'service_type_spellings_after':spellings_after, 'median_distance_km':medians['distance_km'],
 'median_parcel_weight_kg':medians['parcel_weight_kg'], 'median_declared_value_inr':medians['declared_value_inr'],
 'rows_after_hub_merge':len(model), 'nulls_in_features_and_target':int(model[FEATURES+['delivery_hours','is_delayed']].isna().sum().sum()),
 'delayed_count':int(model['is_delayed'].sum()), 'delay_rate':float(model['is_delayed'].mean()),
 'hours_q1':q1, 'hours_q3':q3, 'hours_lower_bound':q1-1.5*iqr, 'hours_upper_bound':q3+1.5*iqr,
 'hours_outlier_count':int(((model['delivery_hours'] < q1-1.5*iqr) | (model['delivery_hours'] > q3+1.5*iqr)).sum()),
 'declared_value_mean_inr':model['declared_value_inr'].mean(), 'declared_value_median_inr':model['declared_value_inr'].median(),
 'log_declared_value_mean':logv.mean(), 'log_declared_value_median':logv.median(),
}
pd.DataFrame([{'metric':k,'value':round(float(v),4) if isinstance(v,(float,np.floating)) else v} for k,v in checks.items()]).to_csv(ROOT/'outputs/part_a_checks.csv',index=False)
model['distance_band'] = pd.cut(model['distance_km'], bins=[0,200,400,600,2000], labels=['0-200','201-400','401-600','601+'], right=True, include_lowest=True)
records=[]
for dim,col in [('service_type','service_type'),('hub_tier','hub_tier'),('distance_band','distance_band')]:
    for group,g in model.groupby(col, observed=False, sort=True):
        if pd.isna(group): continue
        records.append({'dimension':dim,'group':group,'shipments':len(g),'delayed':int(g['is_delayed'].sum()),'delay_rate':round(g['is_delayed'].mean(),4),'mean_hours':round(g['delivery_hours'].mean(),4)})

pd.DataFrame(records).to_csv(ROOT/'outputs/part_a_delay_by_group.csv',index=False)
model.select_dtypes(include=[np.number]).corr().round(4).to_csv(ROOT/'outputs/part_a_correlations.csv',index_label='feature')
print(f"Part A complete: {len(model)} rows; delay rate={model['is_delayed'].mean():.4f}; medians={medians}")

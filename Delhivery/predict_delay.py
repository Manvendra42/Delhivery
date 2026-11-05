from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, clean_shipments, build_features
ROOT=Path(__file__).resolve().parent
raw=pd.read_csv(ROOT/'data/new_shipments.csv')
hubs=pd.read_csv(ROOT/'data/hubs.csv')
med_df=pd.read_csv(ROOT/'models/impute_values.csv')
medians=dict(zip(med_df['column'],med_df['median']))
threshold=float(pd.read_csv(ROOT/'models/delay_threshold.csv')['threshold'].iloc[0])
feature_order=pd.read_csv(ROOT/'models/feature_order.csv')['feature'].tolist()
model=joblib.load(ROOT/'models/delay_model.pkl')
clean, _=clean_shipments(raw,medians=medians)
features=build_features(clean,hubs)
X=features[feature_order]
prob=model.predict_proba(X)[:,1]
scored=pd.DataFrame({'shipment_id':features['shipment_id'],'hub_id':features['hub_id'],'delay_probability':np.round(prob,4),'flagged':(prob>=threshold).astype(int)})
segments=pd.read_csv(ROOT/'outputs/hub_segments.csv')[['hub_id','segment']]
scored=scored.merge(segments,on='hub_id',how='left',validate='many_to_one')
def action(row):
    if row.flagged==0: return 'No action'
    if row.segment=='Escalate': return 'Escalate to hub manager'
    return 'Proactive customer call'
scored['action']=scored.apply(action,axis=1)
scored[['shipment_id','hub_id','segment','delay_probability','flagged','action']].to_csv(ROOT/'outputs/new_shipment_actions.csv',index=False,float_format='%.4f')
print(f"Scored {len(scored)} new shipments; flagged {int(scored.flagged.sum())}")

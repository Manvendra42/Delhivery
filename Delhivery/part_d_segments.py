from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
ROOT=Path(__file__).resolve().parent; out=ROOT/'outputs'; out.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/modelling_table.csv')
profile=df.groupby('hub_id',sort=True).agg(shipments=('shipment_id','count'),mean_delivery_hours=('delivery_hours','mean'),delay_rate=('is_delayed','mean'),mean_distance_km=('distance_km','mean'),mean_n_stops=('n_stops','mean')).reset_index()
profile.round(4).to_csv(out/'hub_profile.csv',index=False)
cols=['mean_delivery_hours','delay_rate','mean_distance_km','mean_n_stops']
scaled=StandardScaler().fit_transform(profile[cols])
D=pairwise_distances(scaled, metric='euclidean')
seps=[]
for k in [2,3,4,5]:
    labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(scaled)
    within=[]; across=[]
    for i in range(len(labels)):
        for j in range(i+1,len(labels)):
            (within if labels[i]==labels[j] else across).append(D[i,j])
    sizes=sorted(pd.Series(labels).value_counts().tolist(),reverse=True)
    seps.append({'K':k,'across_over_within':np.mean(across)/np.mean(within),'cluster_sizes':'-'.join(map(str,sizes))})
pd.DataFrame(seps).round(4).to_csv(out/'part_d_separation.csv',index=False)
labels=KMeans(n_clusters=3,random_state=42,n_init=10).fit_predict(scaled)
profile['cluster']=labels
order=profile.groupby('cluster')['delay_rate'].mean().sort_values().index.tolist()
name_map={order[0]:'Standard',order[1]:'Watch',order[2]:'Escalate'}
profile['segment']=profile['cluster'].map(name_map)
hubs=pd.read_csv(ROOT/'data/hubs.csv')
segments=profile[['hub_id','segment']].merge(hubs[['hub_id','hub_city','hub_tier']],on='hub_id',how='left')
segments=segments[['hub_id','segment','hub_city','hub_tier']]
segments.to_csv(out/'hub_segments.csv',index=False)
segprof=profile.groupby('segment').agg(hubs=('hub_id','count'),mean_delivery_hours=('mean_delivery_hours','mean'),delay_rate=('delay_rate','mean'),mean_distance_km=('mean_distance_km','mean'),mean_n_stops=('mean_n_stops','mean')).reset_index()
segprof['sorter']=segprof['segment'].map({'Standard':0,'Watch':1,'Escalate':2}); segprof=segprof.sort_values('sorter').drop(columns='sorter')
segprof.round(4).to_csv(out/'segment_profile.csv',index=False)
expected={1:'Standard',2:'Watch',3:'Escalate'}
mismatch=segments[segments.apply(lambda r:r['segment']!=expected[int(r['hub_tier'])],axis=1)]
mismatch.to_csv(out/'tier_vs_segment_mismatch.csv',index=False)
print('Part D segmentation complete')

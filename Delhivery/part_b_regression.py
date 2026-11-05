from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES
ROOT=Path(__file__).resolve().parent
out=ROOT/'outputs'; out.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/modelling_table.csv')
X=df[FEATURES]; y=df['delivery_hours']
X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=.3,random_state=42)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=.5,random_state=42)
scaler=StandardScaler().fit(X_train)
tr_scaled=scaler.transform(X_train); te_scaled=scaler.transform(X_test)
pd.DataFrame([{'metric':'train_rows','value':len(X_train)},{'metric':'val_rows','value':len(X_val)},{'metric':'test_rows','value':len(X_test)},{'metric':'train_scaled_max_abs_mean','value':np.abs(tr_scaled.mean(axis=0)).max()},{'metric':'test_scaled_max_abs_mean','value':np.abs(te_scaled.mean(axis=0)).max()}]).to_csv(out/'part_b_checks.csv',index=False,float_format='%.4f')
def metrics(y_true,pred):
    mse=mean_squared_error(y_true,pred)
    return {'MAE':mean_absolute_error(y_true,pred),'MSE':mse,'RMSE':np.sqrt(mse),'R2':r2_score(y_true,pred)}

depths=[1,2,3,4,6,8,12,None]; tree_rows=[]
for d in depths:
    m=DecisionTreeRegressor(max_depth=d,random_state=42).fit(X_train,y_train)
    tree_rows.append({'max_depth':'unlimited' if d is None else d,'train_rmse':metrics(y_train,m.predict(X_train))['RMSE'],'val_rmse':metrics(y_val,m.predict(X_val))['RMSE']})
pd.DataFrame(tree_rows).round(4).to_csv(out/'part_b_tree_depth.csv',index=False)
best_depth=sorted(tree_rows,key=lambda r:(r['val_rmse'], 999 if r['max_depth']=='unlimited' else r['max_depth']))[0]['max_depth']
best_depth=None if best_depth=='unlimited' else int(best_depth)
models={
 'linear_regression':Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),
 'decision_tree':DecisionTreeRegressor(max_depth=best_depth,random_state=42),
 'random_forest':RandomForestRegressor(n_estimators=100,random_state=42),
 'gradient_boosting':GradientBoostingRegressor(n_estimators=100,learning_rate=.1,random_state=42),
}
rows=[]
base=np.full(len(y_val),y_train.mean()); rows.append({'model':'mean_baseline','split':'val',**metrics(y_val,base)})
for name,m in models.items():
    m.fit(X_train,y_train); rows.append({'model':name,'split':'val',**metrics(y_val,m.predict(X_val))})
learning=[]
for lr in [.01,.1,1.0]:
    m=GradientBoostingRegressor(n_estimators=100,learning_rate=lr,random_state=42).fit(X_train,y_train)
    mm=metrics(y_val,m.predict(X_val)); learning.append({'learning_rate':lr,'val_rmse':mm['RMSE'],'val_r2':mm['R2']})
pd.DataFrame(learning).round(4).to_csv(out/'part_b_gb_learning_rate.csv',index=False)
ranked=sorted(models.items(),key=lambda kv:metrics(y_val,kv[1].predict(X_val))['RMSE'])
X_tv=pd.concat([X_train,X_val]); y_tv=pd.concat([y_train,y_val]); cv=KFold(n_splits=5,shuffle=True,random_state=42)
cvrows=[]
for name,m in ranked[:2]:
    scores=cross_val_score(m,X_tv,y_tv,cv=cv)
    cvrows.append({'model':name,'mean_r2':scores.mean(),'std_r2':scores.std()})
pd.DataFrame(cvrows).round(4).to_csv(out/'part_b_cv.csv',index=False)
final_name=ranked[0][0]; final=models[final_name]
rows.append({'model':final_name,'split':'test',**metrics(y_test,final.predict(X_test))})
pd.DataFrame(rows).round(4).to_csv(out/'part_b_metrics.csv',index=False)
print('Part B complete. Best validation model:',final_name,'depth:',best_depth)

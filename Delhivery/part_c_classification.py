from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS
warnings.filterwarnings('ignore', category=RuntimeWarning)
ROOT=Path(__file__).resolve().parent; out=ROOT/'outputs'; out.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/modelling_table.csv'); X=df[FEATURES]; y=df['is_delayed'].astype(int)
X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=.5,random_state=42,stratify=y_temp)
def score_row(name,yt,yp):
    tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
    return {'model':name,'TN':tn,'FP':fp,'FN':fn,'TP':tp,'accuracy':accuracy_score(yt,yp),'precision':precision_score(yt,yp,zero_division=0),'recall':recall_score(yt,yp,zero_division=0),'f1':f1_score(yt,yp,zero_division=0)}
rows=[score_row('blind_all_on_time',y_val,np.zeros(len(y_val),dtype=int))]
log=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression())]).fit(X_train,y_train)
rows.append(score_row('logistic_regression',y_val,log.predict(X_val)))
depth_rows=[]
for d in [2,3,4,5,6,8,10]:
    m=DecisionTreeClassifier(criterion='gini',max_depth=d,random_state=42).fit(X_train,y_train)
    depth_rows.append((d,f1_score(y_val,m.predict(X_val),zero_division=0),m))
best_depth=sorted(depth_rows,key=lambda r:(-r[1],r[0]))[0][0]
tree=next(r[2] for r in depth_rows if r[0]==best_depth)
rows.append(score_row(f'decision_tree_depth_{best_depth}',y_val,tree.predict(X_val)))
rf=RandomForestClassifier(n_estimators=100,random_state=42).fit(X_train,y_train)
rows.append(score_row('random_forest_default',y_val,rf.predict(X_val)))
pd.DataFrame(rows).round(4).to_csv(out/'part_c_models.csv',index=False)
# Signal diagnostics: train-only forest and ANOVA F scores.
all_rf=RandomForestClassifier(n_estimators=100,random_state=42).fit(X_train,y_train)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fvals,_=f_classif(X_train,y_train)
score_df=pd.DataFrame({'feature':FEATURES,'f_score':np.nan_to_num(fvals,nan=0.0),'rf_importance':all_rf.feature_importances_})
score_df['f_rank']=score_df['f_score'].rank(method='min',ascending=False).astype(int)
score_df['is_noise']=score_df['feature'].isin(NOISE_COLUMNS)
score_df.sort_values(['f_rank','feature']).round(4).to_csv(out/'part_c_feature_scores.csv',index=False)
select_rows=[]
for k in [5,7,10,12,17]:
    p=Pipeline([('selector',SelectKBest(f_classif,k=k)),('model',RandomForestClassifier(n_estimators=100,random_state=42))]).fit(X_train,y_train)
    select_rows.append({'k':k,'val_f1':f1_score(y_val,p.predict(X_val),zero_division=0)})
pd.DataFrame(select_rows).round(4).to_csv(out/'part_c_selectk.csv',index=False)
selector=SelectKBest(f_classif,k=7).fit(X_train,y_train)
Xt=selector.transform(X_train); Xv=selector.transform(X_val)
param_grid={'n_estimators':[50,100,150],'max_depth':[4,6,8],'min_samples_leaf':[1,5,10]}
grid=GridSearchCV(RandomForestClassifier(random_state=42),param_grid,cv=5).fit(Xt,y_train)
rand=RandomizedSearchCV(RandomForestClassifier(random_state=42),param_grid,n_iter=9,cv=5,random_state=42).fit(Xt,y_train)
def tune_row(name,candidates,folds,fits,params,bestcv,model):
    pred=model.predict(Xv)
    return {'search':name,'candidates':candidates,'cv_folds':folds,'total_fits':fits,'best_params':str(params),'best_cv_accuracy':bestcv,'val_f1':f1_score(y_val,pred,zero_division=0)}
deforest=RandomForestClassifier(n_estimators=100,random_state=42,min_samples_leaf=1).fit(Xt,y_train)
tuning=[tune_row('grid_search',27,5,135,grid.best_params_,grid.best_score_,grid.best_estimator_),tune_row('randomized_search',9,5,45,rand.best_params_,rand.best_score_,rand.best_estimator_),tune_row('default_forest_same_7_features',1,0,1,{'n_estimators':100,'max_depth':'unlimited','min_samples_leaf':1},np.nan,deforest)]
pd.DataFrame(tuning).round(4).to_csv(out/'part_c_tuning.csv',index=False)
# Highest validation F1; ties prefer default, then grid, then randomized.
priority={'default_forest_same_7_features':0,'grid_search':1,'randomized_search':2}
chosen=sorted(tuning,key=lambda r:(-r['val_f1'],priority[r['search']]))[0]
chosen_name=chosen['search']; chosen_params=chosen['best_params']
if chosen_name=='grid_search': params=grid.best_params_
elif chosen_name=='randomized_search': params=rand.best_params_
else: params={'n_estimators':100,'max_depth':None,'min_samples_leaf':1}
final=Pipeline([('selector',SelectKBest(f_classif,k=7)),('model',RandomForestClassifier(random_state=42,**params))]).fit(X_train,y_train)
val_prob=final.predict_proba(X_val)[:,1]
threshold_rows=[]
thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for t in thresholds:
    threshold_rows.append({'threshold':t,**{k:v for k,v in score_row('x',y_val,(val_prob>=t).astype(int)).items() if k!='model'}})
pd.DataFrame(threshold_rows).round(4).to_csv(out/'part_c_thresholds.csv',index=False)
qualified=[r for r in threshold_rows if r['recall']>=.80]
threshold=max([r['threshold'] for r in qualified]) if qualified else .1
# Test is evaluated once, after configuration and threshold are frozen.
test_prob=final.predict_proba(X_test)[:,1]
testrow=score_row('final',y_test,(test_prob>=threshold).astype(int))
pd.DataFrame([{'threshold':threshold,**{k:v for k,v in testrow.items() if k!='model'}}]).round(4).to_csv(out/'part_c_final_test.csv',index=False)
pd.DataFrame([{'selector_k':7,'n_estimators':params['n_estimators'],'max_depth':'unlimited' if params.get('max_depth') is None else params.get('max_depth'),'min_samples_leaf':params['min_samples_leaf'],'threshold':threshold,'chosen_from':chosen_name}]).to_csv(out/'part_c_final_config.csv',index=False)
print(f'Part C complete: depth={best_depth}, chosen={chosen_name}, threshold={threshold}')

#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

ROOT=Path('/data-1/tmp/verl_agent_scratch/math_stage123_analysis_20260723')
train=pd.read_csv(ROOT/'training_history.csv')
val=pd.read_csv(ROOT/'effective_step_curves.csv')
data=pd.read_csv(ROOT/'training_data_step_features.csv')

offsets={
'b0-stage1':0,'b01-stage1':0,'b0-stage1-control':40,'b01-stage1-control':40,
'b0-stage2-nokl':40,'b0-stage2-m2kl':40,'b01-stage2-nokl':40,'b01-stage2-m2kl':40,
'b0-stage3-nokl-model1':60,'b0-stage3-nokl-model2':60,'b0-stage3-m2kl-model1':60,'b0-stage3-m2kl-model2':60,
'b01-stage3-nokl-model1':60,'b01-stage3-nokl-model2':60,'b01-stage3-m2kl-model1':60,'b01-stage3-m2kl-model2':60}
train['effective_step']=train['training/global_step']+train.id.map(offsets)
train['block_end']=((train['training/global_step']-1)//5+1)*5
train['effective_block_end']=train.block_end+train.id.map(offsets)
agg=train.groupby(['id','block_end','effective_block_end'],as_index=False).agg({
'wdl_sft/correct_ratio':'mean','critic/rewards/mean':'mean','response_length/mean':'mean','response_length/clip_ratio':'mean',
'actor/grad_norm':'mean','actor/pg_loss':'mean','actor/wdl_sft_loss_total':'mean'})
agg.to_csv(ROOT/'training_history_5step_windows.csv',index=False)

# Join validation changes with preceding training windows.
v=val.sort_values(['id','local_step']).copy()
v['delta_mean3']=v.groupby('id').macro_mean3.diff()
v['delta_pass3']=v.groupby('id').macro_pass3.diff()
joined=v.merge(agg,left_on=['id','local_step'],right_on=['id','block_end'],how='left')
joined.to_csv(ROOT/'validation_training_window_join.csv',index=False)

focus=joined[joined.id.isin(['b0-stage1-control','b01-stage1-control','b0-stage3-nokl-model2','b0-stage3-m2kl-model2','b01-stage3-nokl-model2','b01-stage3-m2kl-model2'])].dropna(subset=['delta_mean3'])
corrows=[]
for x in ['wdl_sft/correct_ratio','critic/rewards/mean','response_length/mean','response_length/clip_ratio','actor/grad_norm','actor/pg_loss']:
 for y in ['macro_mean3','delta_mean3']:
  a=focus[[x,y]].dropna()
  pr=pearsonr(a[x],a[y]); sr=spearmanr(a[x],a[y])
  corrows.append({'x':x,'y':y,'n':len(a),'pearson_r':pr.statistic,'pearson_p':pr.pvalue,'spearman_r':sr.statistic,'spearman_p':sr.pvalue})
pd.DataFrame(corrows).to_csv(ROOT/'training_metric_correlations.csv',index=False)

# Data features joined to Stage3 block and validation delta; same data rows apply to control + Stage3.
stage3_data=data[data.phase=='stage3'].copy()
stage3_data['block_end']=((stage3_data.local_step-1)//5+1)*5
data5=stage3_data.groupby('block_end',as_index=False).agg({'mean_level':'mean','level4plus_rate':'mean','level5_rate':'mean','mean_question_chars':'mean','mean_solution_chars':'mean','subject_entropy':'mean'})
model2=focus[focus.id.str.contains('stage3') & focus.id.str.contains('model2')]
perblock=model2.groupby('block_end',as_index=False).agg({'delta_mean3':'mean','wdl_sft/correct_ratio':'mean','response_length/mean':'mean'})
data_join=data5.merge(perblock,on='block_end')
data_join.to_csv(ROOT/'stage3_data_metric_join.csv',index=False)
cor=[]
for x in ['mean_level','level4plus_rate','level5_rate','mean_question_chars','mean_solution_chars','subject_entropy']:
 for y in ['delta_mean3','wdl_sft/correct_ratio','response_length/mean']:
  pr=pearsonr(data_join[x],data_join[y]); sr=spearmanr(data_join[x],data_join[y])
  cor.append({'x':x,'y':y,'n':len(data_join),'pearson_r':pr.statistic,'pearson_p':pr.pvalue,'spearman_r':sr.statistic,'spearman_p':sr.pvalue})
pd.DataFrame(cor).to_csv(ROOT/'stage3_data_metric_correlations.csv',index=False)

# Figure 1: validation and training dynamics.
fig,axes=plt.subplots(2,2,figsize=(18,11),sharex=True)
colors={'b0-stage1-control':'#111827','b01-stage1-control':'#6b7280','b0-stage3-nokl-model2':'#2563eb','b0-stage3-m2kl-model2':'#dc2626','b01-stage3-nokl-model2':'#16a34a','b01-stage3-m2kl-model2':'#9333ea'}
labels={k:k for k in colors}
for run,c in colors.items():
 p=val[val.id==run].sort_values('effective_step')
 axes[0,0].plot(p.effective_step,p.macro_mean3*100,marker='o',label=labels[run],color=c)
 p=agg[agg.id==run].sort_values('effective_block_end')
 axes[0,1].plot(p.effective_block_end,p['wdl_sft/correct_ratio']*100,marker='o',label=labels[run],color=c)
 axes[1,0].plot(p.effective_block_end,p['response_length/mean'],marker='o',label=labels[run],color=c)
 axes[1,1].plot(p.effective_block_end,p['response_length/clip_ratio']*100,marker='o',label=labels[run],color=c)
for ax in axes.ravel():
 ax.axvspan(75,85,color='#f59e0b',alpha=.12,label='observed transition 75-85')
 ax.axvline(80,color='#9ca3af',ls=':',lw=1)
 ax.grid(alpha=.25); ax.set_xlim(40,100)
axes[0,0].set_title('Math-7 macro mean@3'); axes[0,0].set_ylabel('%')
axes[0,1].set_title('Training correct ratio (5-step mean)'); axes[0,1].set_ylabel('%')
axes[1,0].set_title('Mean rollout response length (5-step mean)'); axes[1,0].set_ylabel('tokens')
axes[1,1].set_title('Rollout truncation/clip ratio (5-step mean)'); axes[1,1].set_ylabel('%')
for ax in axes[1]: ax.set_xlabel('Effective step')
handles,lab=axes[0,0].get_legend_handles_labels(); fig.legend(handles,lab,loc='lower center',ncol=3,fontsize=9)
fig.suptitle('Math Stage123: validation jump and training dynamics',fontsize=16)
fig.tight_layout(rect=(0,0.08,1,.96)); fig.savefig(ROOT/'math_stage123_training_dynamics.png',dpi=180); plt.close(fig)

# Figure 2: fixed data windows.
fig,axes=plt.subplots(2,2,figsize=(18,11),sharex=True)
d=stage3_data.sort_values('local_step')
for ax in axes.ravel(): ax.axvspan(16,25,color='#f59e0b',alpha=.15); ax.grid(alpha=.25); ax.set_xlim(1,40)
axes[0,0].plot(d.local_step,d.mean_level,marker='o'); axes[0,0].set_title('Mean Hendrycks MATH level per batch')
axes[0,1].plot(d.local_step,d.level4plus_rate*100,marker='o',label='Level >=4'); axes[0,1].plot(d.local_step,d.level5_rate*100,marker='o',label='Level 5'); axes[0,1].legend(); axes[0,1].set_title('Difficulty mix per 64-row batch'); axes[0,1].set_ylabel('%')
axes[1,0].plot(d.local_step,d.mean_question_chars,marker='o',label='question chars'); axes[1,0].plot(d.local_step,d.mean_solution_chars,marker='o',label='solution chars'); axes[1,0].legend(); axes[1,0].set_title('Reference text length')
subject_cols=[c for c in d.columns if c.startswith('subject_') and c.endswith('_rate')]
for c in subject_cols: axes[1,1].plot(d.local_step,d[c]*100,label=c[len('subject_'):-len('_rate')].replace('_',' '),alpha=.8)
axes[1,1].legend(fontsize=8,ncol=2); axes[1,1].set_title('Subject composition per batch'); axes[1,1].set_ylabel('%')
for ax in axes[1]: ax.set_xlabel('Stage3 local step (fixed data order)')
fig.suptitle('Stage3 fixed-order training data windows; shaded = local steps 16-25',fontsize=16)
fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(ROOT/'math_stage123_data_window_analysis.png',dpi=180); plt.close(fig)

print(pd.DataFrame(corrows).to_string(index=False))
print('\nDATA JOIN\n',data_join.to_string(index=False))
print('\nDATA CORR\n',pd.DataFrame(cor).to_string(index=False))

#!/usr/bin/env python3
import csv, json, re
from pathlib import Path
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

RUNS={
'b0-stage1':'/data-1/tmp/verl_agent_scratch/math_stage123_step20_lr1e6_mem055_entropyoff_launch_20260720T091917Z/recipe/on_policy_wdl_sft/staged_v1/MATH-B0_STAGE1-QWEN3-1P7B-V1_1784539170.log',
'b01-stage1':'/data-1/tmp/verl_agent_scratch/math_stage123_step20_lr1e6_mem055_entropyoff_launch_20260720T091917Z/recipe/on_policy_wdl_sft/staged_v1/MATH-B01_STAGE1-QWEN3-1P7B-V1_1784549110.log',
'b0-stage1-control':'/data-1/tmp/verl_agent_scratch/math_stage123_step20_lr1e6_mem055_entropyoff_launch_20260720T091917Z/recipe/on_policy_wdl_sft/staged_v1/MATH-B0_STAGE1_CONTROL-QWEN3-1P7B-V1_1784558992.log',
'b01-stage1-control':'/data-1/tmp/verl_agent_scratch/math_stage123_step20_lr1e6_mem055_entropyoff_launch_20260720T091917Z/recipe/on_policy_wdl_sft/staged_v1/MATH-B01_STAGE1_CONTROL-QWEN3-1P7B-V1_1784573526.log',
'b0-stage2-nokl':'/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T013450Z-resume-b0-stage2/stage123_queue.log',
'b0-stage2-m2kl':'/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T062805Z-resume-b0-stage2-m2kl/run_attempt_logs/b0-stage2-m2kl.attempt-1.log',
'b01-stage2-nokl':'/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T062805Z-resume-b0-stage2-m2kl/run_attempt_logs/b01-stage2-nokl.attempt-1.log',
'b01-stage2-m2kl':'/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T062805Z-resume-b0-stage2-m2kl/run_attempt_logs/b01-stage2-m2kl.attempt-2.log',
'b0-stage3-nokl-model1':'/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T121155Z-resume-b01-stage3/run_attempt_logs/b0-stage3-nokl-model1.attempt-1.log',
'b0-stage3-nokl-model2':'/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T121155Z-resume-b01-stage3/run_attempt_logs/b0-stage3-nokl-model2.attempt-1.log',
'b0-stage3-m2kl-model1':'/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T121155Z-resume-b01-stage3/run_attempt_logs/b0-stage3-m2kl-model1.attempt-1.log',
'b0-stage3-m2kl-model2':'/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T121155Z-resume-b01-stage3/run_attempt_logs/b0-stage3-m2kl-model2.attempt-1.log',
'b01-stage3-nokl-model1':'/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260722T013739Z-resume-b01-stage3-after-cleanup/run_attempt_logs/b01-stage3-nokl-model1.attempt-1.log',
'b01-stage3-nokl-model2':'/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260722T013739Z-resume-b01-stage3-after-cleanup/run_attempt_logs/b01-stage3-nokl-model2.attempt-1.log',
'b01-stage3-m2kl-model1':'/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260722T013739Z-resume-b01-stage3-after-cleanup/run_attempt_logs/b01-stage3-m2kl-model1.attempt-1.log',
'b01-stage3-m2kl-model2':'/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260722T013739Z-resume-b01-stage3-after-cleanup/run_attempt_logs/b01-stage3-m2kl-model2.attempt-1.log',
}
KEYS=['training/global_step','wdl_sft/correct_ratio','critic/rewards/mean','critic/score/mean','response_length/mean','response_length_non_aborted/mean','response_length/clip_ratio','response/aborted_ratio','actor/grad_norm','actor/pg_loss','actor/wdl_sft_loss_total','actor/wdl_sft_loss_positive','actor/wdl_sft_loss_negative','actor/lr','perf/mfu/actor']

def wandb_dir(log):
 text=Path(log).read_text(errors='replace')
 matches=re.findall(r'Run data is saved locally in ([^\s]+)',text)
 if not matches: raise RuntimeError(f'no wandb path {log}')
 return Path(matches[-1])

def rows(path):
 files=list(path.glob('run-*.wandb'))
 assert len(files)==1,(path,files)
 ds=DataStore(); ds.open_for_scan(str(files[0])); out=[]
 while True:
  data=ds.scan_data()
  if data is None: break
  pb=wandb_internal_pb2.Record(); pb.ParseFromString(data)
  if pb.WhichOneof('record_type')!='history': continue
  item={}
  for x in pb.history.item:
   key='/'.join(x.nested_key) if len(x.nested_key)>1 else x.nested_key[0]
   try: item[key]=json.loads(x.value_json)
   except: item[key]=x.value_json
  if 'training/global_step' in item:
   out.append({k:item.get(k) for k in KEYS})
 return out

allrows=[]
for run,log in RUNS.items():
 d=wandb_dir(log)
 for r in rows(d): allrows.append({'id':run,'wandb_dir':str(d),**r})
out='/data-1/tmp/verl_agent_scratch/math_stage123_analysis_20260723/training_history.csv'
with open(out,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['id','wandb_dir']+KEYS); w.writeheader(); w.writerows(allrows)
print(out,len(allrows))

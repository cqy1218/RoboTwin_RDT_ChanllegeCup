#!/bin/bash

policy_name=RDT  # 明确使用RDT模型
task_name=${1}
task_config=${2}
model_name=${3}    # RDT训练时的模型配置名（对应训练时的${model_name}）
checkpoint_id=${4} # RDT模型检查点ID（如10000表示checkpoint-10000）
seed=${5}
gpu_id=${6}

# RDT专用参数
rdt_step=30        # RDT的预测步长（与训练时一致）
use_ee_control=true # RDT默认输出末端执行器位姿

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mUsing GPU: ${gpu_id}\033[0m"
echo -e "\033[33mRDT Model: ${model_name}, Checkpoint: ${checkpoint_id}\033[0m"

cd ../.. # 切换到项目根目录

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --checkpoint_id ${checkpoint_id} \
    --rdt_step ${rdt_step} \
    --use_ee_control ${use_ee_control} \
    --policy_name ${policy_name}
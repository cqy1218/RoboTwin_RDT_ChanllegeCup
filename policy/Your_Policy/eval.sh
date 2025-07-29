#!/bin/bash

policy_name=RDT  # ��ȷʹ��RDTģ��
task_name=${1}
task_config=${2}
model_name=${3}    # RDTѵ��ʱ��ģ������������Ӧѵ��ʱ��${model_name}��
checkpoint_id=${4} # RDTģ�ͼ���ID����10000��ʾcheckpoint-10000��
seed=${5}
gpu_id=${6}

# RDTר�ò���
rdt_step=30        # RDT��Ԥ�ⲽ������ѵ��ʱһ�£�
use_ee_control=true # RDTĬ�����ĩ��ִ����λ��

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mUsing GPU: ${gpu_id}\033[0m"
echo -e "\033[33mRDT Model: ${model_name}, Checkpoint: ${checkpoint_id}\033[0m"

cd ../.. # �л�����Ŀ��Ŀ¼

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
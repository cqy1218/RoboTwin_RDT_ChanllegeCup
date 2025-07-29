# import packages and module here
import sys, os
from .model import RDT  # ������RDT�ࣨ���ж����ɸ���ʵ�ʵ�����

# ��ȡ��ǰ�ļ�����·���͸�Ŀ¼·�������ں���ģ���ļ���λ
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def encode_obs(observation):  # Post-Process Observation
    """
    ��ԭʼ�۲����ݽ���Ԥ������ȡ�ؼ������ֶΣ�
        - RGBͼ�����飨head / right / left��
        - agent_pos: ��ǰ�����˹ؽ�״̬
        - instruction: ��ѡ��Ȼ����ָ�Ĭ�Ͽգ�
    """
    observation["agent_pos"] = observation["joint_action"]["vector"]
    observation["input_rgb_arr"] = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    observation["instruction"] = observation.get("instruction", "")
    return observation


def get_model(usr_args):  # keep
    """
    ģ�ͼ��������������ò�����ʼ��RDTģ��
    """
    model_name = usr_args["ckpt_setting"]
    checkpoint_id = usr_args["checkpoint_id"]
    left_arm_dim, right_arm_dim, rdt_step = (
        usr_args["left_arm_dim"],
        usr_args["right_arm_dim"],
        usr_args["rdt_step"],
    )

    # 2025.0713 debug���޸���Ȩ���ļ�Ŀ���ַ
    model_path = os.path.join(
        parent_directory,
        f"checkpoints/{model_name}/checkpoint-{checkpoint_id}/model.safetensors"
    )

    rdt = RDT(
        model_path=model_path,
        task_name=usr_args["task_name"],
        left_arm_dim=left_arm_dim,
        right_arm_dim=right_arm_dim,
        pred_step=rdt_step
    )
    return rdt


def eval(TASK_ENV, model, observation):
    """
    ģ��������ջ����������̣�
        - ����۲�
        - ��ȡ����ָ��
        - ��ʼ��ģ�ʹ��ڣ����״Σ�
        - ����������
        - ����ִ�ж��������´���
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    input_rgb_arr = obs["input_rgb_arr"]
    input_state = obs["agent_pos"]

    if model.observation_window is None:
        # ��һ������������ʾ + ��ʼ���۲ⴰ��
        model.set_language_instruction(instruction)
        model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()  # �ɻ�����ʷ��������ಽ����

    for action in actions:
        TASK_ENV.take_action(action)  # Ĭ��ʹ�ùؽڿ��ƣ�qpos��
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        input_rgb_arr = obs["input_rgb_arr"]
        input_state = obs["agent_pos"]
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    """
    ���ģ�ͻ��棨����ʷ���ڣ��Ա������һ������
    """
    # ע�⣺ԭʵ�������󣬽�����RDT����������Ϊ reset_observation_windows
    if hasattr(model, "reset_observation_windows"):
        model.reset_observation_windows()
    elif hasattr(model, "reset_obsrvationwindows"):
        model.reset_obsrvationwindows()
    elif hasattr(model, "clear_action_buffer"):
        model.clear_action_buffer()
    elif hasattr(model, "obs_cache"):
        model.obs_cache.clear()

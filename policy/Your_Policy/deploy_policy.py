# import packages and module here
import sys, os
from .model import RDT  # 仅导入RDT类（如有多个类可根据实际调整）

# 获取当前文件绝对路径和父目录路径，用于后续模型文件定位
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def encode_obs(observation):  # Post-Process Observation
    """
    对原始观测数据进行预处理，提取关键输入字段：
        - RGB图像数组（head / right / left）
        - agent_pos: 当前机器人关节状态
        - instruction: 可选自然语言指令（默认空）
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
    模型加载器：根据配置参数初始化RDT模型
    """
    model_name = usr_args["ckpt_setting"]
    checkpoint_id = usr_args["checkpoint_id"]
    left_arm_dim, right_arm_dim, rdt_step = (
        usr_args["left_arm_dim"],
        usr_args["right_arm_dim"],
        usr_args["rdt_step"],
    )

    # 2025.0713 debug：修改了权重文件目标地址
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
    模型推理与闭环交互主流程：
        - 编码观测
        - 获取语言指令
        - 初始化模型窗口（若首次）
        - 推理动作序列
        - 按序执行动作并更新窗口
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    input_rgb_arr = obs["input_rgb_arr"]
    input_state = obs["agent_pos"]

    if model.observation_window is None:
        # 第一次设置语言提示 + 初始化观测窗口
        model.set_language_instruction(instruction)
        model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()  # 可基于历史窗口推理多步动作

    for action in actions:
        TASK_ENV.take_action(action)  # 默认使用关节控制（qpos）
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        input_rgb_arr = obs["input_rgb_arr"]
        input_state = obs["agent_pos"]
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    """
    清除模型缓存（如历史窗口）以便进行新一轮任务
    """
    # 注意：原实现名有误，建议在RDT类中重命名为 reset_observation_windows
    if hasattr(model, "reset_observation_windows"):
        model.reset_observation_windows()
    elif hasattr(model, "reset_obsrvationwindows"):
        model.reset_obsrvationwindows()
    elif hasattr(model, "clear_action_buffer"):
        model.clear_action_buffer()
    elif hasattr(model, "obs_cache"):
        model.obs_cache.clear()

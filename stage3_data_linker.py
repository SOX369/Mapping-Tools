import os
import json
import random
from typing import List, Dict, Tuple

"""
阶段三模块：数据模块链接
- 匹配网络结构与数据库中的算子数据文件（权重、输出数据）
- 生成第一层输入数据（随机128位二进制数据）
- 按层链接各任务所需的权重数据和输出数据
- 处理层间数据流：每层输入数据来自上一层输出数据
- 生成数据地址映射表（data_addresses.json）
- 将数据模块与任务指令文件合并，输出包含控制+任务+数据的完整配置
"""

# 常量定义
SEPARATOR = "1" * 128
SEPARATOR_LINES = [SEPARATOR + "\n"] * 5


def load_network_structure(network_path: str) -> List[Dict]:
    """加载网络结构配置（统一kernel为元组）"""
    with open(network_path, "r", encoding="utf-8") as f:
        network = json.load(f)
    for layer in network:
        if "kernel" in layer:
            layer["kernel"] = tuple(layer["kernel"])
    return network


def calculate_input_lines(first_layer: Dict) -> int:
    """计算第一层输入数据所需行数：n = ⌈in_H / 8⌉ * in_W * in_channels"""
    # 注意：此函数假设第一层为卷积或池化，对于FC层作为首层的情况需要额外适配
    if first_layer['operator'] in ['Conv', 'Pool']:
        in_H, in_W, in_channels = first_layer["in_H"], first_layer["in_W"], first_layer["in_channels"]
        # 向上取整
        return ((in_H + 7) // 8) * in_W * in_channels
    # 为FC层作为首层预留逻辑
    elif first_layer['operator'] == 'FC':
        # 这里的计算逻辑需要根据硬件如何接收一维向量来确定
        # 暂时假设每行128bit，每个特征8bit
        in_features = first_layer['in_features']
        return (in_features + 15) // 16
    return 0


def generate_random_input(n: int) -> List[str]:
    """生成n行128bit随机01二进制数据（作为输入数据块）"""
    return [''.join(random.choices(['0', '1'], k=128)) + "\n" for _ in range(n)]


def match_conv_db_operator(layer, target_out_channels, operators):
    """在数据库中匹配一个卷积算子（逻辑与stage1一致）"""
    for op in operators:
        if op["operator_type"] != "Conv": continue
        if op["input_channels"] != layer["in_channels"]: continue
        if op["kernel_size"] != list(layer["kernel"]): continue
        if op["stride"] != [layer["stride"], layer["stride"]]: continue
        if op.get("padding", [0, 0]) != [layer.get("padding", 0), layer.get("padding", 0)]: continue
        if op["output_channels"] != target_out_channels: continue
        if op["input_tensor_shape"][0] != layer["in_W"]: continue
        if op["input_tensor_shape"][1] != layer["in_H"]: continue
        if op["output_tensor_shape"][0] != layer["out_W"]: continue
        if op["output_tensor_shape"][1] != layer["out_H"]: continue
        return op
    return None


def match_pool_db_operator(layer, operators):
    """在数据库中匹配一个池化算子（逻辑与stage1一致）"""
    for op in operators:
        if op["operator_type"] != "Pool": continue
        if op["input_channels"] != layer["in_channels"]: continue
        if op["kernel_size"] != list(layer["kernel"]): continue
        if op["stride"] != [layer["stride"], layer["stride"]]: continue
        if op.get("input_tensor_shape", [0, 0, 0])[0] != layer["in_W"]: continue
        if op.get("input_tensor_shape", [0, 0, 0])[1] != layer["in_H"]: continue
        if op.get("output_tensor_shape", [0, 0, 0])[0] != layer["out_W"]: continue
        if op.get("output_tensor_shape", [0, 0, 0])[1] != layer["out_H"]: continue
        if op.get("output_channels", 0) != layer["out_channels"]: continue
        return op
    return None

# ================= FC SUPPORT ADDED START =================
def match_fc_db_operator(layer, target_out_features, operators):
    """在数据库中匹配一个全连接算子（逻辑与stage1一致）"""
    for op in operators:
        if op["operator_type"] != "FC": continue
        if op["in_features"][0] != layer["in_features"]: continue
        if op["out_features"][0] != target_out_features: continue
        if op["isPrevFC"] != layer["isPrevFC"]: continue
        return op
    return None
# ================= FC SUPPORT ADDED END =================


def read_db_operators(db_root: str) -> List[Dict]:
    """读取数据库中所有算子信息（含路径），类似stage1中的read_operator_library"""
    operators = []
    for op_dir in os.listdir(db_root):
        op_path = os.path.join(db_root, op_dir)
        if not os.path.isdir(op_path): continue
        info_path = os.path.join(op_path, "info.json")
        if not os.path.exists(info_path): continue
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                op_info = json.load(f)
            # 记录算子路径，用于后续读取权重和输出文件
            op_info["op_path"] = op_path
            operators.append(op_info)
        except Exception as e:
            print(f"警告：读取算子信息失败 {op_path}，错误：{str(e)}")
    return operators


def link_layer_data(layer: Dict, layer_idx: int, db_operators: List[Dict], current_line: int, task_counter: int) -> \
Tuple[List[str], List[Dict], Dict, int, int]:
    """
    链接一层中所有任务的数据（权重/输出），记录地址，并返回生成的数据内容。
    """
    data_content = []
    task_records = []
    layer_addresses = {}

    # --- 统一确定该层的任务数量 ---
    task_count = 1
    if layer["operator"] == "Conv":
        task_count = (layer["out_channels"] + 9) // 10
        total_out = layer.get("out_channels", 0)
    # ================= FC SUPPORT ADDED START =================
    elif layer["operator"] == "FC":
        task_count = (layer["out_features"] + 9) // 10
        total_out = layer.get("out_features", 0)
    # ================= FC SUPPORT ADDED END =================

    # --- 步骤1: 统一收集该层所有任务的权重和输出数据 ---
    weight_lines_all = []
    output_lines_all = []
    task_op_info = []  # 用于存储每个任务匹配到的算子信息

    for task_idx in range(task_count):
        # 匹配数据库中的算子
        matched_op = None
        if layer["operator"] == "Conv":
            current_out = min(10, total_out - task_idx * 10)
            matched_op = match_conv_db_operator(layer, current_out, db_operators)
        elif layer["operator"] == "Pool":
            matched_op = match_pool_db_operator(layer, db_operators)
        # ================= FC SUPPORT ADDED START =================
        elif layer["operator"] == "FC":
            current_out = min(10, total_out - task_idx * 10)
            matched_op = match_fc_db_operator(layer, current_out, db_operators)
        # ================= FC SUPPORT ADDED END =================

        if not matched_op:
            error_details = (f"层{layer_idx}任务{task_idx + 1}未找到匹配算子\n"
                             f"网络层信息：{json.dumps(layer, indent=2)}\n")
            raise FileNotFoundError(error_details)

        op_path = matched_op["op_path"]
        with open(os.path.join(op_path, "info.json"), 'r', encoding='utf-8') as f:
            op_info = json.load(f)
        task_op_info.append(op_info)

        # 读取权重数据（卷积层和全连接层）
        if layer["operator"] in ["Conv", "FC"]:
            weight_path = os.path.join(op_path, "weight_data.txt")
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"权重文件缺失：{weight_path}")
            with open(weight_path, "r", encoding="utf-8") as f:
                weight_lines = [line if line.endswith("\n") else line + "\n" for line in f.readlines()]
            if len(weight_lines) != op_info["weight_data"]:
                print(
                    f"警告：层{layer_idx}任务{task_idx + 1}的权重文件行数({len(weight_lines)})与info.json中记录的行数({op_info['weight_data']})不一致。")
            weight_lines_all.extend(weight_lines)

        # 读取输出数据
        output_path = os.path.join(op_path, "output_data.txt")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"输出数据文件缺失：{output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            output_lines = [line if line.endswith("\n") else line + "\n" for line in f.readlines()]
        if len(output_lines) != op_info["output_data"]:
            print(
                f"警告：层{layer_idx}任务{task_idx + 1}的输出文件行数({len(output_lines)})与info.json中记录的行数({op_info['output_data']})不一致。")
        output_lines_all.extend(output_lines)

    # --- 步骤2: 将收集到的数据块写入内容列表，并计算地址 ---
    # 写入权重数据块
    weight_start_addr = 0
    if layer["operator"] in ["Conv", "FC"]:
        weight_start_addr = current_line
        data_content.extend(weight_lines_all)
        current_line += len(weight_lines_all)
        data_content.extend(SEPARATOR_LINES)
        current_line += 5

    # 写入输出数据块
    output_start_addr = current_line
    data_content.extend(output_lines_all)
    current_line += len(output_lines_all)
    data_content.extend(SEPARATOR_LINES)
    current_line += 5

    # --- 步骤3: 为该层的每个任务分别计算并填充地址映射 ---
    weight_offset = 0
    output_offset = 0
    for task_idx in range(task_count):
        op_info = task_op_info[task_idx]
        task_key = f"{task_counter + task_idx + 1}_task"

        task_weight_addr = 0
        weight_lines = 0
        if layer["operator"] in ["Conv", "FC"]:
            task_weight_addr = weight_start_addr + weight_offset
            weight_lines = op_info.get("weight_data", 0)
            weight_offset += weight_lines

        task_output_addr = output_start_addr + output_offset
        output_lines = op_info.get("output_data", 0)
        output_offset += output_lines

        layer_addresses[task_key] = {
            "inputData_addr": 0,  # 临时占位，后续统一更新
            "weightData_addr": task_weight_addr,
            "outputData_addr": task_output_addr,
            "weight_lines": weight_lines,
            "output_lines": output_lines
        }
        # 同时记录用于日志打印的信息
        task_records.append({
            "layer": layer_idx,
            "task": task_counter + task_idx + 1,
            "operator_type": layer["operator"],
            "weight_start": task_weight_addr,
            "output_start": task_output_addr
        })

    return data_content, task_records, layer_addresses, current_line, task_counter + task_count


def process_data_module(network: List[Dict], task_file_path: str, db_operators: List[Dict]) -> Tuple[
    List[str], Dict, List[Dict]]:
    """处理整个数据模块的生成：生成输入数据 + 链接各层数据 + 生成地址映射"""
    # 读取任务指令文件内容（作为基础）
    with open(task_file_path, "r", encoding="utf-8") as f:
        task_content = f.readlines()
    task_lines_count = len(task_content)

    # 初始化数据内容（从任务指令末尾开始）
    data_content = []
    # 添加任务模块与数据模块的分隔符
    data_content.extend(SEPARATOR_LINES)
    current_line = task_lines_count + 5

    # 生成第一层输入数据（整个网络唯一的随机输入）
    first_layer = network[0]
    input_lines_needed = calculate_input_lines(first_layer)
    input_data = generate_random_input(input_lines_needed)

    # 记录输入数据地址并添加到数据内容中
    input_start_addr = current_line
    data_content.extend(input_data)
    current_line += input_lines_needed
    data_content.extend(SEPARATOR_LINES)
    current_line += 5

    # 按层处理数据
    all_addresses = {}  # 存储最终的地址映射表
    all_records = []  # 存储用于日志打印的记录
    prev_layer_output_addr = input_start_addr  # 第一层的输入是随机生成的输入数据
    task_counter = 0

    for layer_idx, layer in enumerate(network, 1):
        print(f"处理层 {layer_idx}：{layer['operator']}（输入数据起始地址：{prev_layer_output_addr}）")
        if layer['operator'] in ['Conv', 'Pool']:
            print(f"  层信息：in_W={layer['in_W']}, in_H={layer['in_H']}, in_channels={layer['in_channels']}")
            print(f"          out_W={layer['out_W']}, out_H={layer['out_H']}, out_channels={layer['out_channels']}")
            if 'kernel' in layer:
                print(f"          kernel={layer['kernel']}, stride={layer['stride']}, padding={layer.get('padding', 0)}")
        elif layer['operator'] == 'FC':
             print(f"  层信息：in_features={layer['in_features']}, out_features={layer['out_features']}, isPrevFC={layer['isPrevFC']}")

        # 链接当前层所有任务的数据（权重+输出）
        layer_data, task_records, layer_addresses, current_line, task_counter = link_layer_data(
            layer, layer_idx, db_operators, current_line, task_counter)

        data_content.extend(layer_data)
        all_records.extend(task_records)

        # 核心逻辑：更新当前层所有任务的输入地址，使其指向上一层的输出地址
        for task_key in layer_addresses:
            layer_addresses[task_key]["inputData_addr"] = prev_layer_output_addr

        all_addresses[f"{layer_idx}_layer"] = layer_addresses

        # 为下一层准备输入：本层的输出地址成为下一层的输入地址
        # 对于多任务层，所有任务共享一个输出地址块的起始地址
        first_task_key_in_layer = sorted(layer_addresses.keys(), key=lambda k: int(k.split('_')[0]))[0]
        prev_layer_output_addr = layer_addresses[first_task_key_in_layer]["outputData_addr"]

    # 返回合并后的完整文件内容和地址信息
    return task_content + data_content, all_addresses, all_records


def print_data_records(records: List[Dict], addresses: Dict):
    """打印数据链接记录和地址映射表（日志）"""
    print("\n==== 数据模块链接记录 ====")
    # 按顺序打印，方便核对
    for layer_key in sorted(addresses.keys(), key=lambda k: int(k.split('_')[0])):
        sorted_tasks = sorted(addresses[layer_key].items(), key=lambda item: int(item[0].split('_')[0]))
        for task_key, addr_info in sorted_tasks:
            layer_num = layer_key.split('_')[0]
            task_num = task_key.split('_')[0]
            print(f"层 {layer_num} 任务 {task_num}:")
            print(f"  输入起始行: {addr_info['inputData_addr']}")
            if addr_info['weight_lines'] > 0:
                print(f"  权重起始行: {addr_info['weightData_addr']}")
            print(f"  输出起始行: {addr_info['outputData_addr']}\n")

    print("==== 数据地址映射表 ====")
    print("data_addresses = {")
    for layer, tasks in addresses.items():
        print(f"  {layer}: {{")
        sorted_tasks = sorted(tasks.items(), key=lambda item: int(item[0].split('_')[0]))
        for task, addr in sorted_tasks:
            print(f"    {task}: {addr},")
        print(f"  }},")
    print("}")


def link_data_module(control_task_file, full_output_file, network_path, db_root, data_address_output_file):
    """
    执行阶段三：链接数据模块并生成数据地址映射表
    """
    print("=" * 20 + " 阶段三：链接数据模块 " + "=" * 20)
    # 验证数据库目录是否存在
    if not os.path.exists(db_root):
        raise FileNotFoundError(f"数据库目录不存在：{os.path.abspath(db_root)}")

    network = load_network_structure(network_path)
    db_operators = read_db_operators(db_root)
    if not db_operators:
        raise ValueError(f"在数据文件库 {db_root} 中未找到有效的算子")

    # 执行数据处理核心逻辑
    full_content, data_addresses, all_records = process_data_module(
        network, control_task_file, db_operators)

    # 合并任务指令与数据模块，写入完整文件
    with open(full_output_file, "w", encoding="utf-8") as f:
        f.writelines(full_content)

    # 保存地址映射为JSON
    with open(data_address_output_file, "w", encoding="utf-8") as f:
        json.dump(data_addresses, f, indent=2, ensure_ascii=False)

    # 打印日志
    print_data_records(all_records, data_addresses)
    print(f"\n数据模块处理完成，输出文件：{full_output_file}")
    print(f"地址映射已保存：{data_address_output_file}")
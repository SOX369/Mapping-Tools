import os
import json

"""
阶段一模块：任务指令划分与任务地址对齐
- 加载网络结构配置文件，识别各层参数（卷积、池化、全连接等）
- 读取算子库中的二进制指令配置，匹配网络层与对应算子
- 按层划分任务：
    - 卷积层按输出通道数划分（每10个通道一个任务）
    - 全连接层按输出特征数划分（每10个特征一个任务）
    - 池化层固定1个任务
- 生成原始任务指令配置文件（包含128位分隔符）
- 进行地址对齐处理（按256的倍数对齐各任务起始地址）
- 输出两个文件：原始版本和地址对齐版本
"""


# 常量定义
SEPARATOR = "1" * 128  # 128bit全1分隔符
SEPARATOR_LINES = [SEPARATOR] * 5  # 任务间固定5行分隔符


def load_network_structure(network_path):
    """加载网络结构配置文件"""
    with open(network_path, "r", encoding="utf-8") as f:
        network = json.load(f)
    # 统一格式，将kernel列表转换为元组，方便后续匹配
    for layer in network:
        if "kernel" in layer:
            layer["kernel"] = tuple(layer["kernel"])
    return network


def read_operator_library(library_path):
    """读取算子库中所有算子信息（含路径）"""
    operators = []
    # 遍历算子库目录下的每个子目录（每个子目录代表一个算子）
    for op_dir in os.listdir(library_path):
        op_path = os.path.join(library_path, op_dir)
        if not os.path.isdir(op_path):
            continue
        # 读取算子配置信息（info.json）
        info_path = os.path.join(op_path, "info.json")
        if not os.path.exists(info_path):
            continue
        with open(info_path, "r", encoding="utf-8") as f:
            op_info = json.load(f)
        # 记录算子路径，用于后续读取激励文件
        op_info["op_path"] = op_path
        operators.append(op_info)
    return operators


def match_conv_operator(layer, target_out_channels, operators):
    """匹配卷积算子（按所有相关字段）"""
    for op in operators:
        # 基础匹配条件：算子类型、输入通道数、kernel/stride/padding
        if op["operator_type"] != "Conv": continue
        if op["input_channels"] != layer["in_channels"]: continue
        if op["kernel_size"] != list(layer["kernel"]): continue
        if op["stride"] != [layer["stride"], layer["stride"]]: continue
        if op.get("padding", [0, 0]) != [layer.get("padding", 0), layer.get("padding", 0)]: continue
        # 任务划分的核心：输出通道数精确匹配
        if op["output_channels"] != target_out_channels: continue
        # 精确匹配输入输出张量尺寸
        if op["input_tensor_shape"][0] != layer["in_W"]: continue
        if op["input_tensor_shape"][1] != layer["in_H"]: continue
        if op["output_tensor_shape"][0] != layer["out_W"]: continue
        if op["output_tensor_shape"][1] != layer["out_H"]: continue
        return op
    return None


def match_pool_operator(layer, operators):
    """匹配池化算子（按所有相关字段）"""
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
def match_fc_operator(layer, target_out_features, operators):
    """匹配全连接算子"""
    for op in operators:
        if op["operator_type"] != "FC": continue
        # 匹配输入特征数、isPrevFC标志 和 目标输出特征数
        if op["in_features"][0] != layer["in_features"]: continue
        if op["out_features"][0] != target_out_features: continue
        if op["isPrevFC"] != layer["isPrevFC"]: continue
        return op
    return None


# ================= FC SUPPORT ADDED END =================

def read_operator_excitation(op_path):
    """读取算子激励文件（op_jili.txt）内容"""
    excite_path = os.path.join(op_path, "op_jili.txt")
    with open(excite_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f.readlines()]


def generate_original_task_file(network, operators, output_path):
    """生成原始任务指令配置文件（含任务划分日志）"""
    original_lines = []
    global_task_idx = 1  # 全局任务计数器，跨层累计

    # 遍历网络结构中的每一层
    for layer_idx, layer in enumerate(network, 1):
        print(f"处理层 {layer_idx}: {layer}")

        # 卷积层：按输出通道划分任务
        if layer["operator"] == "Conv":
            total_out = layer["out_channels"]
            # 计算任务数，向上取整（例如64通道 -> (64+9)//10 = 7个任务）
            task_count = (total_out + 9) // 10
            print(f"  卷积层任务划分：共需 {task_count} 次任务（总输出通道 {total_out}）")
            print(f"  任务范围：第 {global_task_idx} 到第 {global_task_idx + task_count - 1} 次任务")

            # 为每个划分出的任务匹配算子
            for task_idx in range(task_count):
                # 计算当前任务需要处理的输出通道数（通常是10，最后一个任务可能小于10）
                current_out = min(10, total_out - task_idx * 10)
                matched_op = match_conv_operator(layer, current_out, operators)
                if not matched_op:
                    error_msg = (
                        f"未找到匹配的卷积算子：\n"
                        f"  算子类型：Conv，目标输出通道：{current_out}\n"
                        f"  输入：in_W={layer['in_W']}, in_H={layer['in_H']}, in_channels={layer['in_channels']}\n"
                        f"  输出：out_W={layer['out_W']}, out_H={layer['out_H']}\n"
                        f"  kernel={layer['kernel']}, stride={layer['stride']}, padding={layer.get('padding', 0)}"
                    )
                    raise FileNotFoundError(error_msg)
                # 读取并写入算子激励
                excite_lines = read_operator_excitation(matched_op["op_path"])
                original_lines.extend(excite_lines)
                original_lines.extend(SEPARATOR_LINES)
                global_task_idx += 1

        # 池化层：固定为1次任务
        elif layer["operator"] == "Pool":
            task_count = 1
            print(f"  池化层任务划分：共需 {task_count} 次任务")
            print(f"  任务范围：第 {global_task_idx} 到第 {global_task_idx + task_count - 1} 次任务")

            matched_op = match_pool_operator(layer, operators)
            if not matched_op:
                error_msg = (
                    f"未找到匹配的池化算子：\n"
                    f"  算子类型：Pool\n"
                    f"  输入：in_W={layer['in_W']}, in_H={layer['in_H']}, in_channels={layer['in_channels']}\n"
                    f"  输出：out_W={layer['out_W']}, out_H={layer['out_H']}, out_channels={layer['out_channels']}\n"
                    f"  kernel={layer['kernel']}, stride={layer['stride']}"
                )
                raise FileNotFoundError(error_msg)
            # 读取并写入算子激励
            excite_lines = read_operator_excitation(matched_op["op_path"])
            original_lines.extend(excite_lines)
            original_lines.extend(SEPARATOR_LINES)
            global_task_idx += 1

        # ================= FC SUPPORT ADDED START =================
        elif layer["operator"] == "FC":
            total_out_features = layer["out_features"]
            # 按输出特征数划分任务，每10个为一次任务
            task_count = (total_out_features + 9) // 10
            print(f"  全连接层任务划分：共需 {task_count} 次任务（总输出特征 {total_out_features}）")
            print(f"  任务范围：第 {global_task_idx} 到第 {global_task_idx + task_count - 1} 次任务")

            for task_idx in range(task_count):
                current_out = min(10, total_out_features - task_idx * 10)
                matched_op = match_fc_operator(layer, current_out, operators)
                if not matched_op:
                    error_msg = (
                        f"未找到匹配的全连接算子：\n"
                        f"  算子类型：FC，目标输出特征：{current_out}\n"
                        f"  输入特征：{layer['in_features']}\n"
                        f"  isPrevFC: {layer['isPrevFC']}"
                    )
                    raise FileNotFoundError(error_msg)

                excite_lines = read_operator_excitation(matched_op["op_path"])
                original_lines.extend(excite_lines)
                original_lines.extend(SEPARATOR_LINES)
                global_task_idx += 1
        # ================= FC SUPPORT ADDED END =================

    # 写入原始文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(original_lines) + "\n")
    print(f"原始总任务指令配置文件已生成: {output_path}")
    return original_lines


def find_tasks_in_original(original_lines):
    """
    在原始文件内容中识别任务边界，任务的结束标志是连续5行全'1'的分隔符。
    """
    tasks = []
    current_start = 0
    i = 0
    while i < len(original_lines):
        # 跳过文件或任务开头可能存在的全'1'行
        while i < len(original_lines) and original_lines[i] == SEPARATOR:
            i += 1
        if i >= len(original_lines):
            break
        current_start = i

        # 寻找连续5行全'1'作为任务结束的标志
        consecutive = 0
        j = i
        while j < len(original_lines):
            if original_lines[j] == SEPARATOR:
                consecutive += 1
                if consecutive >= 5:
                    task_end = j - 4  # 任务内容不包括这5行分隔符
                    tasks.append((current_start, task_end))
                    print(f"找到任务 {len(tasks)}: 行 {current_start + 1} 到 {task_end}, 共 {task_end - current_start} 行")
                    i = j + 1
                    break
            else:
                consecutive = 0
            j += 1
        else:
            # 处理文件末尾的最后一个任务（可能没有5行分隔符结尾）
            task_end = len(original_lines)
            while task_end > current_start and original_lines[task_end - 1] == SEPARATOR:
                task_end -= 1
            if current_start < task_end:
                tasks.append((current_start, task_end))
                print(f"找到任务 {len(tasks)}: 行 {current_start + 1} 到 {task_end}, 共 {task_end - current_start} 行")
            i = j
    return tasks


def generate_aligned_task_file(tasks, original_lines, output_path):
    """根据找到的任务边界，生成地址对齐的文件"""
    aligned_lines = []
    current_line = 0  # 当前行号，也代表地址

    print(f"在原始文件中找到 {len(tasks)} 个任务，开始进行地址对齐...")
    for task_idx, (start, end) in enumerate(tasks):
        # 对齐处理：除了第一个任务，其他任务的起始地址都必须是256的倍数
        if task_idx > 0:
            # 计算下一个256倍数的地址
            target_address = ((current_line + 255) // 256) * 256
            # 计算需要填充的行数
            padding = target_address - current_line
            if padding > 0:
                aligned_lines.extend([SEPARATOR] * padding)
                current_line += padding
                print(f"任务 {task_idx + 1}: 添加了 {padding} 行全1分隔符，从第 {current_line + 1} 行开始，地址为 {current_line}")
        else:
            print(f"任务 {task_idx + 1}: 从第 {current_line + 1} 行开始，地址为 {current_line}")

        # 写入任务内容
        task_lines = original_lines[start:end]
        aligned_lines.extend(task_lines)
        current_line += len(task_lines)
        print(f"  任务 {task_idx + 1} 写入了 {len(task_lines)} 行指令")

    # 保存对齐后的文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(aligned_lines) + "\n")
    print(f"地址对齐的总任务指令配置文件已生成: {output_path}")


def generate_task_instructions(network_path, library_path, original_output, aligned_output):
    """
    执行阶段一：生成原始和地址对齐的任务指令文件。
    """
    print(f"=" * 20 + " 阶段一：生成任务指令 " + "=" * 20)
    # 加载配置
    network = load_network_structure(network_path)
    operators = read_operator_library(library_path)
    # 生成原始任务文件
    original_lines = generate_original_task_file(network, operators, original_output)
    # 从原始文件内容中识别任务边界
    tasks = find_tasks_in_original(original_lines)

    # 生成地址对齐的文件
    generate_aligned_task_file(tasks, original_lines, aligned_output)
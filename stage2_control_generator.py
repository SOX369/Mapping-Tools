import json

"""
阶段二模块：控制信息与FIFO管理
- 从地址对齐的任务指令文件中提取任务边界和地址信息
- 生成FIFO队列管理信息（包含任务起始地址和指令条数）
- 创建1536行控制器指令配置（前512行为总控指令，513行开始为FIFO信息）
- 将控制信息与任务指令配置合并成完整文件
- 生成并保存任务地址映射表（task_addresses.json）
"""

# 全局常量：总控制器指令
total_controller_instructions = [
    "10001010111000000000000000000100111010110001011100000000000000001000100011100000000000000000101111100110011101001010110110000000",
    "10000110000100110000000000000000100001100011010000000000000000001000101001000000000000000000001011100010000100011001000000000000",
    "10110000111001110000000000000001110000001110000000000000000100101000101011100000000000000000010011101000000110001011100000000000",
    "11000011000000000000000000001100110100000000000000000000000001001011010000000000000000000000000010110100000000000000000000000000",
    "10110100000000000000000000000000101101000000000000000000000000001011010000000000000000000000000011111100000000000000000000000000"
]
SEPARATOR = "1" * 128


def load_network_structure(network_path: str) -> list:
    """加载网络结构配置文件"""
    with open(network_path, "r", encoding="utf-8") as f:
        network = json.load(f)
    for layer in network:
        if "kernel" in layer:
            layer["kernel"] = tuple(layer["kernel"])
    return network


def get_task_counts_per_layer(network: list) -> list:
    """根据网络结构计算每层的任务数量"""
    task_counts = []
    for layer in network:
        if layer["operator"] == "Conv":
            # 卷积层按输出通道划分任务，每10个通道一个任务
            task_counts.append((layer["out_channels"] + 9) // 10)
        # ================= FC SUPPORT ADDED START =================
        elif layer["operator"] == "FC":
            # 全连接层按输出特征数划分任务，每10个特征一个任务
            task_counts.append((layer["out_features"] + 9) // 10)
        # ================= FC SUPPORT ADDED END =================
        else:
            # 其他算子（如Pool）固定为1个任务
            task_counts.append(1)
    return task_counts


def find_tasks_in_aligned_file(task_lines: list) -> list:
    """
    在地址对齐的文件内容中查找每个任务的边界。
    由于文件已对齐，每个任务块由非分隔符行构成，块之间由分隔符行隔开。
    """
    task_info = []
    i = 0
    while i < len(task_lines):
        # 跳过任务间的填龧分隔符
        while i < len(task_lines) and task_lines[i] == SEPARATOR:
            i += 1
        if i >= len(task_lines):
            break

        # 记录任务的起始行
        task_start = i
        # 寻找任务的结束行（即下一个分隔符行）
        j = i
        while j < len(task_lines) and task_lines[j] != SEPARATOR:
            j += 1
        task_end = j
        # 记录任务的（起始行号，指令条数）
        task_info.append((task_start, task_end - task_start))
        i = j
    return task_info


def generate_control_module(aligned_task_file, control_task_output_file, network_path, task_address_output_file):
    """
    执行阶段二：添加控制信息和FIFO管理。
    """
    print("=" * 20 + " 阶段二：生成控制模块 " + "=" * 20)

    # 1. 读取地址对齐后的任务指令文件
    with open(aligned_task_file, "r", encoding="utf-8") as f:
        # 读取所有行并去除首尾空白，忽略空行
        task_lines = [line.strip() for line in f.readlines() if line.strip()]

    # 2. 重新分析任务指令，记录每个任务的起始行号和指令条数
    task_info = find_tasks_in_aligned_file(task_lines)
    print(f"检测到 {len(task_info)} 个任务")

    # 3. 加载网络结构，用于验证任务总数并将任务映射到对应的网络层
    network = load_network_structure(network_path)
    task_counts_per_layer = get_task_counts_per_layer(network)
    print(f"从网络结构获取到 {len(task_counts_per_layer)} 层，每层任务数: {task_counts_per_layer}")

    # 验证从文件中检测到的任务数是否与根据网络结构计算出的任务数相符
    total_expected_tasks = sum(task_counts_per_layer)
    if len(task_info) != total_expected_tasks:
        print(f"警告: 检测到的任务数({len(task_info)})与网络结构预期的任务数({total_expected_tasks})不匹配")

    # 4. 生成任务地址映射表 (task_addresses.json)
    task_addresses = {}
    current_layer = 1
    tasks_in_current_layer = 0
    for idx, (start, count) in enumerate(task_info):
        # 计算任务在最终文件中的绝对行号和地址
        # 最终文件 = 1536行控制信息 + 任务指令
        final_start_line = start + 1536 + 1  # 行号从1开始计数
        address = final_start_line - 1  # 地址从0开始计数
        print(f"任务 {idx + 1}: 地址对齐文件中第 {start + 1} 行, 最终文件中第 {final_start_line} 行, 地址 {address}, 指令条数 {count}")
        print(f"  地址是否为256倍数: {address % 256 == 0}")

        task_key = f"{idx + 1}_task"

        # 根据每层的任务数，判断当前任务属于哪一层
        if current_layer <= len(task_counts_per_layer) and tasks_in_current_layer >= task_counts_per_layer[
            current_layer - 1]:
            current_layer += 1
            tasks_in_current_layer = 0

        if current_layer > len(task_counts_per_layer):
            layer_key = f"{current_layer}_layer"
            print(f"警告: 任务 {idx + 1} 超出网络结构定义的层数({len(task_counts_per_layer)})")
        else:
            layer_key = f"{current_layer}_layer"

        if layer_key not in task_addresses:
            task_addresses[layer_key] = {}

        # 存储任务的关键信息：实际行号、起始地址、指令数
        task_addresses[layer_key][task_key] = {'actual_line': final_start_line, 'origin_addr': address,
                                               'instruction_nums': count}
        tasks_in_current_layer += 1

    # 5. 生成FIFO信息
    fifo_info = []
    for start, count in task_info:
        actual_start_line = start + 1536 + 1
        # FIFO格式：64位(全0) + 32位(起始地址*16) + 32位(指令数)
        part1 = "0" * 64
        part2 = bin((actual_start_line - 1) * 16)[2:].zfill(32)  # 地址需乘以16
        part3 = bin(count)[2:].zfill(32)
        fifo_info.append(part1 + part2 + part3)

    # 6. 创建1536行的控制器指令配置
    # 修改total_controller_instructions中第一行的第81至第96位为FIFO信息条数
    fifo_count_binary = bin(len(fifo_info))[2:].zfill(16)
    temp_controller_instructions = total_controller_instructions[:]  # 创建副本以防修改全局变量
    temp_controller_instructions[0] = temp_controller_instructions[0][:80] + fifo_count_binary + \
                                      temp_controller_instructions[0][96:]

    # 组装完整的1536行控制块
    control_instructions = []
    control_instructions.extend(temp_controller_instructions)
    # 填充到512行（总控指令区）
    while len(control_instructions) < 512:
        control_instructions.append(SEPARATOR)
    # 从第513行开始添加FIFO信息
    control_instructions.extend(fifo_info)
    # 继续填充到1536行
    while len(control_instructions) < 1536:
        control_instructions.append(SEPARATOR)

    # 7. 合并控制指令配置和总任务指令配置文件内容
    new_lines = [line + "\n" for line in control_instructions] + [line + "\n" for line in task_lines]

    # 写入新文件
    with open(control_task_output_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"已生成 {control_task_output_file}，包含 {len(task_info)} 个任务的FIFO信息")

    # 8. 保存任务指令映射表为JSON文件
    with open(task_address_output_file, "w", encoding="utf-8") as f:
        json.dump(task_addresses, f, indent=2, ensure_ascii=False)

    # 打印任务指令映射表到终端
    print("\ntask_addresses = {")
    # 按层号和任务号的数值排序后输出，方便查看
    for layer in sorted(task_addresses.keys(), key=lambda x: int(x.split('_')[0])):
        print(f"  {layer}: {{")
        sorted_tasks = sorted(
            task_addresses[layer].items(),
            key=lambda item: int(item[0].split('_')[0])
        )
        for task_key, data in sorted_tasks:
            data_str = ", ".join([f"'{k}': {v}" for k, v in data.items()])
            print(f"    {task_key}: {{{data_str}}},")
        print(f"  }},")
    print("}")
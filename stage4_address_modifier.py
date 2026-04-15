import json

"""
阶段四模块：存储控制配置地址修改
- 加载任务地址和数据地址映射文件
- 解析任务指令中的存储控制器配置（识别011开头的配置行）
- 根据数据类型（输入/权重/输出）和工作模式，修改相应的地址字段
- 将数据地址转换为27位二进制格式，拆分为高14位和低13位
- 更新存储控制器配置中的地址信息，输出最终可执行的激励文件
"""


def load_json_files(task_addresses_file, data_addresses_file):
    """加载任务地址和数据地址映射的JSON文件"""
    with open(task_addresses_file, "r", encoding="utf-8") as f:
        task_addresses = json.load(f)
    with open(data_addresses_file, "r", encoding="utf-8") as f:
        data_addresses = json.load(f)
    return task_addresses, data_addresses


def replace_bits(binary_str, start, end, new_bits):
    """替换二进制字符串的一个切片"""
    return binary_str[:start] + new_bits + binary_str[end + 1:]


def addr_to_27bit_binary(addr):
    """将地址转换为27位二进制，并拆分为高14位和低13位"""
    full_addr = addr * 16  # 地址需要乘以16
    binary_27bit = format(full_addr, '027b')  # 转换为27位二进制字符串
    high_14bit = binary_27bit[:14]  # 高14位
    low_13bit = binary_27bit[14:]  # 低13位
    return high_14bit, low_13bit


def get_task_data_addresses(layer_idx, task_idx, data_addresses):
    """根据层索引和任务索引，从数据地址映射表中获取数据地址"""
    layer_key = f"{layer_idx}_layer"
    task_key = f"{task_idx}_task"
    if layer_key in data_addresses and task_key in data_addresses[layer_key]:
        return data_addresses[layer_key][task_key]
    return None


def modify_task_storage_config(lines, start_line_1_based, task_data_addrs):
    """
    修改单个任务指令块中的存储控制器配置地址字段。
    此函数会直接修改传入的 `lines` 列表。
    """
    # 将1-based的行号转换为0-based的列表索引
    i = start_line_1_based - 1

    # 设定一个扫描范围，假设一个任务的指令不超过180行，以提高效率
    scan_end = min(i + 180, len(lines))

    # 遍历任务指令，寻找 '011' 开头的存储控制器配置块（3行一组）
    while i <= scan_end - 3:
        line1 = lines[i].strip()
        if len(line1) == 128 and line1.startswith('011'):
            line3 = lines[i + 2].strip()

            # 从指令中提取关键字段：数据位宽(dw)和工作模式(work_mode)
            dw = int(line1[23:25], 2)
            work_mode = int(line3[113:115], 2)

            addr_to_use = None
            data_type = "未知"
            # 根据工作模式和数据位宽，判断当前指令对应的数据类型
            if work_mode == 0:  # DDR_TO_MC (从DDR读)
                if dw == 2:
                    addr_to_use = task_data_addrs['inputData_addr']
                    data_type = "输入"
                elif dw == 1:
                    addr_to_use = task_data_addrs['weightData_addr']
                    data_type = "权重"
            elif work_mode == 2:  # MC_TO_DDR (写到DDR)
                if dw == 2:
                    addr_to_use = task_data_addrs['outputData_addr']
                    data_type = "输出"

            # 如果成功匹配到需要修改的地址
            if addr_to_use is not None and addr_to_use >= 0:
                print(f"  修改{data_type}数据配置，地址: {addr_to_use}")
                # 将地址转换为高14位和低13位的二进制
                high_14bit, low_13bit = addr_to_27bit_binary(addr_to_use)

                # 替换第3行指令中的地址字段
                # 修改 initial_addr_height_14bit (bits 50-63)
                modified_line3 = replace_bits(line3, 50, 63, high_14bit)
                # 修改 initial_addr_low_13bit (bits 115-127)
                modified_line3 = replace_bits(modified_line3, 115, 127, low_13bit)

                # 更新列表中的行内容
                lines[i + 2] = modified_line3 + '\n'

                print(f"    原始地址: {addr_to_use}, 乘16后: {addr_to_use * 16}")
                print(f"    27位二进制: {format(addr_to_use * 16, '027b')}")
                print(f"    高14位: {high_14bit}, 低13位: {low_13bit}")

            # 跳过这个已处理的3行指令块
            i += 3
        else:
            i += 1


def modify_final_addresses(input_file, final_output_file, task_addresses_file, data_addresses_file):
    """
    执行阶段四：在最终文件中修改存储控制器的地址
    """
    print("=" * 20 + " 阶段四：修改最终地址 " + "=" * 20)
    print("开始修改存储控制器配置中的地址字段...")

    # 1. 加载任务和数据地址映射文件
    task_addresses, data_addresses = load_json_files(task_addresses_file, data_addresses_file)

    # 2. 读取包含控制、任务和数据的完整配置文件
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 3. 按层和任务遍历，逐个修改地址
    global_task_counter = 1
    # 按层号排序遍历
    for layer_key in sorted(task_addresses.keys(), key=lambda k: int(k.split('_')[0])):
        layer_idx = int(layer_key.split('_')[0])
        print(f"\n处理第{layer_idx}层:")

        # 按任务号排序遍历
        sorted_tasks = sorted(task_addresses[layer_key].items(), key=lambda item: int(item[0].split('_')[0]))

        for task_key, task_info in sorted_tasks:
            task_idx = int(task_key.split('_')[0])

            # 获取任务的起始行号
            actual_line = task_info['actual_line']

            # 获取该任务对应的数据地址
            task_data_addrs = get_task_data_addresses(layer_idx, task_idx, data_addresses)
            if task_data_addrs is None:
                print(f"  警告: 未找到任务{task_idx}的数据地址信息，跳过修改。")
                continue

            print(f"  任务{task_idx} (全局任务{global_task_counter}):")
            print(f"    起始行: {actual_line}")
            print(f"    输入地址: {task_data_addrs['inputData_addr']}")
            print(f"    权重地址: {task_data_addrs['weightData_addr']}")
            print(f"    输出地址: {task_data_addrs['outputData_addr']}")

            # 调用函数，修改当前任务的存储控制器配置
            modify_task_storage_config(lines, actual_line, task_data_addrs)
            global_task_counter += 1

    # 4. 写入修改后的文件
    with open(final_output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n地址修改完成！输出文件: {final_output_file}")


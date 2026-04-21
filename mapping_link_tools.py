import os
import sys
import argparse

# 导入你现有的各个阶段模块（不需要修改这些模块内部代码）
import stage0_onnx_to_json
import stage1_task_generator
import stage2_control_generator
import stage3_data_linker
import stage4_address_modifier


def get_resource_path(relative_path):
    """
    核心路径处理函数：
    获取资源文件的绝对路径。如果被 PyInstaller 打包成 exe，则从临时解压目录 _MEIPASS 读取；
    如果是 Python 源码环境，则从当前目录读取。
    """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 打包后的临时执行目录
        base_path = sys._MEIPASS
    else:
        # 开发时的当前工作目录
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def run_pipeline(network_path, output_dir):
    """运行阶段 1 到阶段 4 的核心映射流水线"""

    # 获取被打包到 exe 内部的算子库和数据库的真实路径
    op_library_path = get_resource_path("Op_Library")
    data_db_root = get_resource_path("Data_Library")

    # 路径校验，防止打包时遗漏
    if not os.path.exists(op_library_path):
        print(f"[报错] 找不到算子库文件夹: {op_library_path}")
        sys.exit(1)
    if not os.path.exists(data_db_root):
        print(f"[报错] 找不到数据文件夹: {data_db_root}")
        sys.exit(1)

    # 在输出文件夹下定义所有的中间及结果文件路径
    original_task_file = os.path.join(output_dir, "1_original_tasks.txt")
    aligned_task_file = os.path.join(output_dir, "1_aligned_tasks.txt")
    control_task_file = os.path.join(output_dir, "2_control_and_tasks.txt")
    task_addresses_json = os.path.join(output_dir, "task_addresses.json")
    full_config_file = os.path.join(output_dir, "3_full_config_with_data.txt")
    data_addresses_json = os.path.join(output_dir, "data_addresses.json")
    final_output_file = os.path.join(output_dir, "final_executable_config.txt")

    try:
        # ======= 阶段一：生成任务指令 =======
        stage1_task_generator.generate_task_instructions(
            network_path=network_path,
            library_path=op_library_path,
            original_output=original_task_file,
            aligned_output=aligned_task_file
        )

        # ======= 阶段二：生成控制模块和FIFO =======
        stage2_control_generator.generate_control_module(
            aligned_task_file=aligned_task_file,
            control_task_output_file=control_task_file,
            network_path=network_path,
            task_address_output_file=task_addresses_json
        )

        # ======= 阶段三：链接数据模块 =======
        stage3_data_linker.link_data_module(
            control_task_file=control_task_file,
            full_output_file=full_config_file,
            network_path=network_path,
            db_root=data_db_root,
            data_address_output_file=data_addresses_json
        )

        # ======= 阶段四：修改最终地址 =======
        stage4_address_modifier.modify_final_addresses(
            input_file=full_config_file,
            final_output_file=final_output_file,
            task_addresses_file=task_addresses_json,
            data_addresses_file=data_addresses_json
        )

        print(f"\n[成功] 处理完成！最终可执行文件已保存至: {final_output_file}")

    except Exception as e:
        print(f"\n[报错] 流程执行异常: {e}")
        sys.exit(1)


def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Mapping Link Tools - 神经网络映射离线工具链")
    parser.add_argument("--output_file", required=True, help="处理结果的输出文件夹名称")
    parser.add_argument("--model", type=str, help="ONNX模型文件路径 (给出此参数则自动转JSON)")
    parser.add_argument("--model_struct_json", type=str, help="模型结构描述文件(json文件)路径")

    args = parser.parse_args()

    # 确保输出目录一定存在
    os.makedirs(args.output_file, exist_ok=True)

    # 2. 根据用户输入的参数决定执行逻辑
    if args.model:
        print(f"[*] 模式: ONNX模型输入 -> {args.model}")
        print(f"[*] 输出文件夹: {args.output_file}")

        # 将 onnx 转换后的 json 临时保存在输出目录下
        json_tmp_path = os.path.join(args.output_file, "network_structure_tmp.json")

        try:
            print("[*] 正在执行 Stage 0: 转换 ONNX 到 JSON...")
            converter = stage0_onnx_to_json.ONNXToNetworkStructure(args.model)
            converter.convert()
            converter.save_to_json(json_tmp_path)
            print(f"[*] Stage 0 转换成功: {json_tmp_path}")
        except Exception as e:
            print(f"[报错] ONNX 转换失败: {e}")
            sys.exit(1)

        # 以生成的 json 继续后续阶段
        run_pipeline(json_tmp_path, args.output_file)

    elif args.model_struct_json:
        print(f"[*] 模式: JSON描述文件输入 -> {args.model_struct_json}")
        print(f"[*] 输出文件夹: {args.output_file}")
        # 直接拿 json 继续后续阶段
        run_pipeline(args.model_struct_json, args.output_file)

    else:
        print("\n[报错] 参数缺失！必须提供 --model 或 --model_struct_json 之一作为输入源。")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


#pyinstaller --noconfirm --onefile --console --name "mapping_link_tools" --add-data "Op_Library;Op_Library" --add-data "Data_Library;Data_Library" mapping_link_tools.py

import os
import stage1_task_generator
import stage2_control_generator
import stage3_data_linker
import stage4_address_modifier


def run_pipeline():

    NETWORK_PATH = "network_structure_output.json"
    OP_LIBRARY_PATH = "Op_Library"
    DATA_DB_ROOT = "Data_Library"
    # 中间及输出文件路径
    OUTPUT_DIR = "pipeline_output"

    # 阶段一输出
    ORIGINAL_TASK_FILE = os.path.join(OUTPUT_DIR, "1_original_tasks.txt")
    ALIGNED_TASK_FILE = os.path.join(OUTPUT_DIR, "1_aligned_tasks.txt")

    # 阶段二输出
    CONTROL_TASK_FILE = os.path.join(OUTPUT_DIR, "2_control_and_tasks.txt")
    TASK_ADDRESSES_JSON = os.path.join(OUTPUT_DIR, "task_addresses.json")

    # 阶段三输出
    FULL_CONFIG_FILE = os.path.join(OUTPUT_DIR, "3_full_config_with_data.txt")
    DATA_ADDRESSES_JSON = os.path.join(OUTPUT_DIR, "data_addresses.json")

    # 阶段四输出
    FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_executable_config.txt")
    # ==================================================

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # 生成任务指令与任务地址对齐
        stage1_task_generator.generate_task_instructions(
            network_path=NETWORK_PATH,
            library_path=OP_LIBRARY_PATH,
            original_output=ORIGINAL_TASK_FILE,
            aligned_output=ALIGNED_TASK_FILE
        )

        # 生成控制模块和FIFO
        stage2_control_generator.generate_control_module(
            aligned_task_file=ALIGNED_TASK_FILE,
            control_task_output_file=CONTROL_TASK_FILE,
            network_path=NETWORK_PATH,
            task_address_output_file=TASK_ADDRESSES_JSON
        )

        # 链接数据模块
        stage3_data_linker.link_data_module(
            control_task_file=CONTROL_TASK_FILE,
            full_output_file=FULL_CONFIG_FILE,
            network_path=NETWORK_PATH,
            db_root=DATA_DB_ROOT,
            data_address_output_file=DATA_ADDRESSES_JSON
        )

        # 修改最终地址
        stage4_address_modifier.modify_final_addresses(
            input_file=FULL_CONFIG_FILE,
            final_output_file=FINAL_OUTPUT_FILE,
            task_addresses_file=TASK_ADDRESSES_JSON,
            data_addresses_file=DATA_ADDRESSES_JSON
        )

        print(f"最终可执行文件位于: {FINAL_OUTPUT_FILE}")

    except Exception as e:
        print(f"\n发生错误，错误详情: {e}")


if __name__ == "__main__":
    run_pipeline()


import customtkinter as ctk
import os
import sys
import threading
import time
import queue
from tkinter import filedialog
import json

# ==============================================================================
# 导入你的功能模块 (Stage 0 - Stage 4)
# ==============================================================================
try:
    import stage0_onnx_to_json
    import stage1_task_generator
    import stage2_control_generator
    import stage3_data_linker
    import stage4_address_modifier
except ImportError as e:
    print(f"警告: 缺少必要的模块文件。\n错误详情: {e}")

# ==============================================================================
# 全局配置 & 颜色定义
# ==============================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# 状态指示灯颜色
COLOR_WAITING = "gray40"  # 灰色：待执行
COLOR_RUNNING = "#3B8ED0"  # 蓝色：执行中 (Theme Blue)
COLOR_SUCCESS = "#2CC985"  # 绿色：已完成
COLOR_ERROR = "#E04F5F"  # 红色：出错

# 控制日志刷新速度的参数
QUEUE_POLL_INTERVAL = 3000


# 日志重定向器，不再直接操作 GUI，而是将文本放入线程安全的队列中。
class BufferedConsoleRedirector:

    def __init__(self, msg_queue):
        self.msg_queue = msg_queue

    def write(self, string):
        if string:
            self.msg_queue.put(string)

    def flush(self):
        pass


class AIChipToolchainApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- 1. 基础窗口设置 ---
        self.title("AXIP Toolchain Generator | AI 芯片工具链可视化")
        self.geometry("1150x750")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 线程通信队列
        self.log_queue = queue.Queue()
        self.is_running = False

        # --- 2. 布局初始化 ---
        self.setup_sidebar()
        self.setup_main_area()

        # 启动日志轮询定时器
        self.check_log_queue()

    def setup_sidebar(self):
        """创建左侧边栏：包含配置输入和进度条"""
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)  # 让底部有弹性空间

        # LOGO区
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AXIP 工具链控制台",
                                       font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 5))

        self.desc_label = ctk.CTkLabel(self.sidebar_frame, text="End-to-End Compiler Pipeline",
                                       text_color="gray70", font=ctk.CTkFont(size=12))
        self.desc_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # 参数输入区
        self.input_fields = {}
        self.create_file_selector(2, "onnx_path", "ONNX 模型文件 (.onnx)", is_file=True,
                                  file_type=[("ONNX Model", "*.onnx")])
        self.create_file_selector(3, "json_path", "网络结构 JSON (可选)", is_file=True, file_type=[("JSON File", "*.json")])
        self.create_file_selector(4, "op_lib", "算子库目录 (Op_Library)", is_file=False)
        self.create_file_selector(5, "data_lib", "数据库目录 (Data_Library)", is_file=False)
        self.create_file_selector(6, "output_dir", "输出产物目录", is_file=False, default_text="pipeline_output")

        # 分割线
        self.separator = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="gray30")
        self.separator.grid(row=7, column=0, padx=20, pady=20, sticky="ew")

        # --- [需求 1] 执行进度面板 ---
        self.create_progress_panel(row=8)

        # 运行按钮
        self.run_button = ctk.CTkButton(self.sidebar_frame, text="启动工具链 (Execute Pipeline)",
                                        height=40, font=ctk.CTkFont(weight="bold"),
                                        command=self.start_pipeline_thread)
        self.run_button.grid(row=9, column=0, padx=20, pady=(10, 30))

    def create_progress_panel(self, row):
        """构建进度展示区域"""
        progress_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        progress_frame.grid(row=row, column=0, padx=20, pady=5, sticky="nsew")

        title = ctk.CTkLabel(progress_frame, text="Waiting Tasks | 执行进度", font=ctk.CTkFont(size=14, weight="bold"),
                             anchor="w")
        title.pack(fill="x", pady=(0, 10))

        # 定义5个阶段的名称
        self.stage_names = [
            "阶段 0: ONNX 模型解析 (可选)",
            "阶段 1: 任务指令生成 (Task Gen)",
            "阶段 2: 控制信息配置 (Control Gen)",
            "阶段 3: 数据模块链接 (Data Linker)",
            "阶段 4: 地址修正 (Address Mod)"
        ]

        self.stage_indicators = []  # 存储指示灯控件引用

        for name in self.stage_names:
            item_frame = ctk.CTkFrame(progress_frame, fg_color="transparent", height=30)
            item_frame.pack(fill="x", pady=2)

            # 圆形指示灯 (使用Button模拟，禁用点击)
            indicator = ctk.CTkButton(item_frame, text="", width=12, height=12, corner_radius=6,
                                      fg_color=COLOR_WAITING, state="disabled", hover=False)
            indicator.pack(side="left", padx=(0, 10))

            # 文本标签
            lbl = ctk.CTkLabel(item_frame, text=name, text_color="gray80", font=ctk.CTkFont(size=12))
            lbl.pack(side="left")

            self.stage_indicators.append(indicator)

    def create_file_selector(self, row, key, label_text, is_file=True, file_type=None, default_text=""):
        frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        frame.grid(row=row, column=0, padx=20, pady=5, sticky="ew")

        lbl = ctk.CTkLabel(frame, text=label_text, anchor="w", font=ctk.CTkFont(size=12))
        lbl.pack(fill="x")

        entry = ctk.CTkEntry(frame, height=28, placeholder_text=default_text)
        if default_text: entry.insert(0, default_text)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        def browse():
            path = filedialog.askopenfilename(filetypes=file_type) if is_file else filedialog.askdirectory()
            if path:
                entry.delete(0, "end")
                entry.insert(0, path)

        btn = ctk.CTkButton(frame, text="..", width=30, height=28, command=browse)
        btn.pack(side="right")
        self.input_fields[key] = entry

    def setup_main_area(self):
        """右侧主日志区域"""
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # 顶部栏
        header = ctk.CTkFrame(self.main_frame, height=50, fg_color="transparent")
        header.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="ew")

        ctk.CTkLabel(header, text="Pipeline Execution Logs", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        ctk.CTkButton(header, text="Clear Logs", width=80, height=28, fg_color="gray40",
                      command=self.clear_log).pack(side="right")

        # 终端文本框
        self.console_textbox = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(family="Consolas", size=13),
                                              activate_scrollbars=True)
        self.console_textbox.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.console_textbox.insert("0.0", "--- Ready to start ---\n")
        self.console_textbox.configure(state="disabled")

    # ==============================================================================
    # 核心逻辑：UI 更新与事件处理
    # ==============================================================================

    def update_stage_status(self, stage_idx, status_color):
        """更新左侧进度指示灯的颜色"""
        if 0 <= stage_idx < len(self.stage_indicators):
            self.stage_indicators[stage_idx].configure(fg_color=status_color)

    def reset_progress_ui(self):
        """重置所有指示灯为灰色"""
        for indicator in self.stage_indicators:
            indicator.configure(fg_color=COLOR_WAITING)

    def clear_log(self):
        self.console_textbox.configure(state="normal")
        self.console_textbox.delete("0.0", "end")
        self.console_textbox.configure(state="disabled")

    def check_log_queue(self):
        """
        [性能优化核心]
        定时器：每隔 QUEUE_POLL_INTERVAL 毫秒，将队列中的所有日志一次性取出并显示。
        这避免了频繁的 GUI 刷新。
        """
        try:
            content_chunk = ""
            # 一次性取出队列中目前所有的文本（非阻塞）
            while not self.log_queue.empty():
                content_chunk += self.log_queue.get_nowait()

            if content_chunk:
                self.console_textbox.configure(state="normal")
                self.console_textbox.insert("end", content_chunk)
                self.console_textbox.see("end")  # 自动滚动到底部
                self.console_textbox.configure(state="disabled")
        except queue.Empty:
            pass
        finally:
            # 重新调度自己，实现循环
            self.after(QUEUE_POLL_INTERVAL, self.check_log_queue)

    def start_pipeline_thread(self):
        if self.is_running: return

        params = {k: v.get() for k, v in self.input_fields.items()}
        if not params["output_dir"]:
            self.log_queue.put("\n[Error] Please specify output directory.\n")
            return

        self.is_running = True
        self.run_button.configure(state="disabled", text="Running...")
        self.reset_progress_ui()
        self.clear_log()

        # 开启后台线程
        t = threading.Thread(target=self.run_pipeline_logic, args=(params,))
        t.daemon = True
        t.start()

    def run_pipeline_logic(self, params):
        """后台线程执行逻辑，通过 callback 更新 UI"""
        old_stdout = sys.stdout
        sys.stdout = BufferedConsoleRedirector(self.log_queue)  # 劫持 print

        output_dir = params["output_dir"]
        network_path = params["json_path"]
        onnx_path = params["onnx_path"]

        # 内部 helper：执行单个阶段的包装器
        def execute_stage(stage_idx, stage_name, func, **kwargs):
            try:
                # 1. 变蓝 (Running)
                self.after(0, lambda: self.update_stage_status(stage_idx, COLOR_RUNNING))
                print(f"\n{'=' * 15} [{stage_name}] Start {'=' * 15}")

                # 2. 执行实际函数
                func(**kwargs)

                # 3. 变绿 (Success)
                self.after(0, lambda: self.update_stage_status(stage_idx, COLOR_SUCCESS))
                return True
            except Exception as e:
                # 4. 变红 (Error)
                self.after(0, lambda: self.update_stage_status(stage_idx, COLOR_ERROR))
                print(f"\n[ERROR in {stage_name}] {str(e)}")
                import traceback
                traceback.print_exc()
                raise e  # 抛出异常中断后续步骤

        try:
            os.makedirs(output_dir, exist_ok=True)

            # --- Stage 0: ONNX Logic ---
            # 如果有ONNX且没JSON，或者用户想重新生成，就跑Stage 0
            should_run_stage0 = bool(onnx_path) and (not network_path or not os.path.exists(network_path))

            if should_run_stage0:
                if not network_path:
                    network_path = os.path.join(output_dir, "network_structure.json")

                def run_s0():
                    converter = stage0_onnx_to_json.ONNXToNetworkStructure(onnx_path)
                    converter.convert()
                    converter.save_to_json(network_path)

                execute_stage(0, "Stage 0: ONNX Parse", run_s0)
            else:
                # 如果没跑Stage0，直接把灯变成绿色(如果已提供json)或保持灰色
                if network_path and os.path.exists(network_path):
                    self.after(0, lambda: self.update_stage_status(0, COLOR_SUCCESS))

            if not network_path or not os.path.exists(network_path):
                raise FileNotFoundError("Missing Network JSON. Please provide ONNX or JSON path.")

            # 定义路径
            original_task = os.path.join(output_dir, "1_original_tasks.txt")
            aligned_task = os.path.join(output_dir, "1_aligned_tasks.txt")
            control_task = os.path.join(output_dir, "2_control_and_tasks.txt")
            task_json = os.path.join(output_dir, "task_addresses.json")
            full_config = os.path.join(output_dir, "3_full_config_with_data.txt")
            data_json = os.path.join(output_dir, "data_addresses.json")
            final_out = os.path.join(output_dir, "final_executable_config.txt")

            # --- Stage 1 ---
            execute_stage(1, "Stage 1: Task Gen",
                          stage1_task_generator.generate_task_instructions,
                          network_path=network_path, library_path=params["op_lib"],
                          original_output=original_task, aligned_output=aligned_task)

            # --- Stage 2 ---
            execute_stage(2, "Stage 2: Control Gen",
                          stage2_control_generator.generate_control_module,
                          aligned_task_file=aligned_task, control_task_output_file=control_task,
                          network_path=network_path, task_address_output_file=task_json)

            # --- Stage 3 ---
            execute_stage(3, "Stage 3: Data Linker",
                          stage3_data_linker.link_data_module,
                          control_task_file=control_task, full_output_file=full_config,
                          network_path=network_path, db_root=params["data_lib"],
                          data_address_output_file=data_json)

            # --- Stage 4 ---
            execute_stage(4, "Stage 4: Address Mod",
                          stage4_address_modifier.modify_final_addresses,
                          input_file=full_config, final_output_file=final_out,
                          task_addresses_file=task_json, data_addresses_file=data_json)

            print(f"\n✅ All stages completed successfully!")
            print(f"Output File: {final_out}")

        except Exception as e:
            print("\n❌ Pipeline Aborted.")

        finally:
            sys.stdout = old_stdout
            self.is_running = False
            self.after(0, lambda: self.run_button.configure(state="normal", text="启动工具链 (Execute Pipeline)"))


if __name__ == "__main__":
    app = AIChipToolchainApp()
    app.mainloop()


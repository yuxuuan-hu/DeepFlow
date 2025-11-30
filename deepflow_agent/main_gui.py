# -*- coding: utf-8 -*-

# --- Standard Library Imports ---
import os
import sys
import signal
import re
import math
import subprocess
from datetime import datetime
from pathlib import Path

# --- Third-Party Data & Math Imports ---
import pandas as pd
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import FuncFormatter

# --- PyQt5 Imports ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit,
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QDialog, QFormLayout, QDialogButtonBox, QLabel,
                             QSplitter, QFrame, QMessageBox)
from PyQt5.QtCore import (QThread, pyqtSignal, QObject, pyqtSlot, Qt, QTimer, 
                          QSettings, QEventLoop)
from PyQt5.QtGui import QTextCursor, QFont

# --- LangChain Imports ---
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor
from langchain_core.tools import StructuredTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction

# --- Custom Module Imports ---
from openfoam_agent import (OpenFOAMRAGAgent, read_file, list_files, write_file,
                            search_and_replace_in_file, insert_line_into_file,
                            delete_lines_from_file, execute_shell_command, 
                            query_log_file, read_csv_file, read_log_tail, send_email)
from plot_data import plot_residuals, plot_flowrate
import log2csv

# --- Configuration ---
DEFAULT_SETTINGS = {
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
    "LANGCHAIN_API_KEY": "YOUR_LANGCHAIN_API_KEY",
    "OPENAI_API_KEY": "YOUR_OPENAI_API_KEY",
    "CUSTOM_API_BASE": "YOUR_CUSTOM_API_BASE",
    "CASE_DIR": "YOUR_CASE_DIRECTORY_PATH",
    # Replace 'icoFoam' with your specific solver, keep the redirection tail
    "SOLVER_COMMAND": "icoFoam > run.log 2>&1 &",
    "RUN_LOG_PATH": "YOUR_CASE_DIRECTORY_PATH/run.log",
    "MODEL": "qwen3-coder-plus"
}

# --- Utility Functions & Classes ---

def simple_read_file(file_path: str) -> str:
    """Reads and returns file content."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def simple_write_file(file_path: str, content: str):
    """Writes content to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

class HtmlFormatter:
    """Formats log messages into HTML for the UI."""
    COLORS = {
        "user": "#00529B", "system": "#4A4A4A", "param": "#483D8B",
        "ui": "#006400", "agent_thought": "#008B8B", "tool": "#C75F00",
        "agent_final": "#0047AB", "confirm": "#E69500", "success": "#228B22",
        "error": "#B22222", "default": "#000000"
    }

    @staticmethod
    def format(text, msg_type):
        color = HtmlFormatter.COLORS.get(msg_type, HtmlFormatter.COLORS["default"])
        style = (f"color: {color}; white-space: pre-wrap; margin: 5px 0; "
                 f"font-size: 30px; font-family: 'DejaVu Sans', 'Noto Color Emoji', sans-serif;")
        
        prefix_map = {
            "user": "👤 <b>[User]:</b>", "system": "⚙️ <b>[System]:</b>",
            "param": "🔧 <b>[Parameters]:</b>", "ui": "🖼️ <b>[UI]:</b>",
            "agent_thought": "🤔 <b>[Agent Thinking]:</b>", "tool": "🛠️ <b>[Executing Tool]:</b>",
            "agent_final": "🤖 <b>[Agent Final Conclusion]:</b>", "confirm": "⚠️ <b>[Confirmation Required]:</b>",
            "success": "✅ <b>[Success]:</b>", "error": "❌ <b>[Error]:</b>",
        }
        
        prefix = prefix_map.get(msg_type, "")
        escaped_text = text.replace('<', '&lt;').replace('>', '&gt;')
        return f'<div style="{style}">{prefix}<br>{escaped_text}</div>'

class UserConfirmationRequired(Exception):
    """Interrupts execution to request user confirmation."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

def confirm_and_execute_wrapper(main_window, **kwargs) -> str:
    """
    Pauses the agent thread to wait for UI confirmation before executing sensitive tools.
    """
    main_window.user_confirmation_response = None
    action_type = kwargs.get('action_type')
    
    # Construct confirmation message based on action
    if action_type == 'write':
        msg = f"Overwrite file `{kwargs['file_path']}`?\nContent Preview:\n---\n{kwargs['content']}\n---\nEnter `yes` to confirm."
    elif action_type == 'search_replace':
        msg = f"Replace in `{kwargs['file_path']}`:\nSearch: `{kwargs['search_string']}`\nReplace: `{kwargs['replace_string']}`\nEnter `yes` to confirm."
    elif action_type == 'insert':
        msg = f"Insert at line {kwargs['line_number']} in `{kwargs['file_path']}`:\nContent:\n---\n{kwargs['content_to_insert']}\n---\nEnter `yes` to confirm."
    elif action_type == 'delete':
        end = kwargs.get('end_line') or kwargs['start_line']
        msg = f"Delete lines {kwargs['start_line']}-{end} in `{kwargs['file_path']}`?\nEnter `yes` to confirm."
    elif action_type == 'execute':
        path = kwargs.get('working_directory') or main_window.case_dir
        msg = f"Execute command in `{path}`:\n`{kwargs['command']}`\nEnter `yes` to confirm."
    else:
        msg = "Unknown operation requiring confirmation."

    # Trigger UI loop
    main_window.confirmation_event_loop = QEventLoop()
    main_window.confirmation_request_signal.emit(msg)
    main_window.confirmation_event_loop.exec_()
    main_window.confirmation_event_loop = None

    if main_window.user_confirmation_response == 'yes':
        main_window.log_signal.emit("User approved. Executing...", 'system')
        try:
            if action_type == 'write':
                return write_file.func(kwargs['file_path'], kwargs['content'])
            elif action_type == 'search_replace':
                return search_and_replace_in_file.func(kwargs['file_path'], kwargs['search_string'], kwargs['replace_string'])
            elif action_type == 'insert':
                return insert_line_into_file.func(kwargs['file_path'], kwargs['line_number'], kwargs['content_to_insert'])
            elif action_type == 'delete':
                return delete_lines_from_file.func(kwargs['file_path'], kwargs['start_line'], kwargs.get('end_line'))
            elif action_type == 'execute':
                output = execute_shell_command.func(command=kwargs['command'], working_directory=kwargs.get('working_directory'))
                main_window.log_signal.emit(f"Command output:\n---\n{output}\n---", 'system')
                return output
        except Exception as e:
            return f"Execution error: {e}"
    else:
        return "User rejected the operation."

# --- Logic & Threading Classes ---

class QtCallbackHandler(QObject, BaseCallbackHandler):
    """LangChain callback handler to bridge agent events to the Qt UI."""
    new_message = pyqtSignal(str, str)

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Handles agent thought process logging."""
        thought = action.log.strip()
        if thought:
            reason = thought.split("responded:", 1)[1].strip() if "responded:" in thought else thought.split("Invoking:")[0].strip()
            
            if reason:
                self.main_window.last_agent_thought = reason
                self.new_message.emit(reason, "agent_thought")
            else:
                # Default reasons for standard tools
                tool_name = action.tool
                dangerous_tools = {write_file.name, search_and_replace_in_file.name, 
                                   insert_line_into_file.name, delete_lines_from_file.name, 
                                   execute_shell_command.name}
                
                if tool_name not in dangerous_tools:
                    tool_purposes = {
                        read_file.name: "read file content.",
                        list_files.name: "check file structure.",
                        query_log_file.name: "query log info.",
                        'openfoam_knowledge_retriever': "query knowledge base.",
                    }
                    purpose = tool_purposes.get(tool_name, f"execute `{tool_name}`.")
                    msg = f"Next step: call `{tool_name}` to {purpose}"
                    self.main_window.last_agent_thought = msg
                    self.new_message.emit(msg, "agent_thought")

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Handles tool execution logging."""
        tool_name = serialized.get("name", "Unknown tool")
        dangerous_tools = {write_file.name, search_and_replace_in_file.name, 
                           insert_line_into_file.name, delete_lines_from_file.name, 
                           execute_shell_command.name}
        
        if tool_name in dangerous_tools:
            self.new_message.emit(f"Preparing dangerous operation: `{tool_name}`. Waiting for confirmation...", "tool")
        else:
            msgs = {
                read_file.name: f"Reading file: {input_str}",
                list_files.name: f"Listing dir: {input_str}",
                query_log_file.name: f"Querying log: {input_str}",
                read_csv_file.name: f"Reading CSV: {input_str}",
                read_log_tail.name: f"Reading log tail: {input_str}",
                send_email.name: "Error detected, sending email...",
                'openfoam_knowledge_retriever': "Querying knowledge base...",
            }
            self.new_message.emit(msgs.get(tool_name, f"Calling `{tool_name}`..."), "tool")

class AgentWorker(QThread):
    """Worker thread for running the LangChain agent."""
    agent_finished = pyqtSignal(dict)
    confirmation_required = pyqtSignal(str) 
    agent_error = pyqtSignal(str)

    def __init__(self, agent_executor, instruction, chat_history, callback_handler):
        super().__init__()
        self.agent_executor = agent_executor
        self.instruction = instruction
        self.chat_history = chat_history
        self.callback_handler = callback_handler

    def run(self):
        try:
            result = self.agent_executor.invoke(
                {"input": self.instruction, "chat_history": self.chat_history},
                {"callbacks": [self.callback_handler]}
            )
            self.agent_finished.emit(result)
        except UserConfirmationRequired as e:
            self.confirmation_required.emit(e.message)
        except Exception as e:
            self.agent_error.emit(f'Critical Agent Error: {e}')

# --- UI Widgets ---

class SettingsDialog(QDialog):
    """Dialog for editing application settings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = QSettings("DeepFlow", "AgentApp")
        self.layout = QFormLayout(self)
        self.inputs = {}
        
        for key, _ in DEFAULT_SETTINGS.items():
            self.inputs[key] = QLineEdit(self)
            self.layout.addRow(QLabel(f"{key}:"), self.inputs[key])

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.save_settings)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)
        self.load_settings()

    def load_settings(self):
        for key, default in DEFAULT_SETTINGS.items():
            self.inputs[key].setText(str(self.settings.value(key, default)))

    def save_settings(self):
        for key, widget in self.inputs.items():
            self.settings.setValue(key, widget.text())
        self.accept()

class ParameterWidget(QWidget):
    """Widget for viewing and editing OpenFOAM parameters."""
    log_message = pyqtSignal(str)
    
    def __init__(self, case_dir, parent=None):
        super().__init__(parent)
        self.case_dir = case_dir
        self.param_editors = {}
        self.init_ui()
        self.refresh_parameters()

    def init_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        # Define parameters to track
        params = {
            "endTime": ("End Time (s)", "N/A"),
            "deltaT": ("Time Step (s)", "N/A"),
            "writeInterval": ("Write Interval", "N/A"),
            "rpm1": ("Fan 1 RPM", "N/A"),
            "rpm2": ("Fan 2 RPM", "N/A"),
            "p_relax": ("Pressure (P) Relax", "N/A"),
            "U_relax": ("Velocity (U) Relax", "N/A"),
            "k_relax": ("TKE (k) Relax", "N/A"),
            "omega_relax": ("Omega Relax", "N/A"),
        }

        font = QFont()
        font.setPointSize(11)
        
        for key, (label, default) in params.items():
            editor = QLineEdit(default)
            editor.setFont(font)
            self.param_editors[key] = editor
            lbl = QLabel(label)
            lbl.setFont(font)
            form_layout.addRow(lbl, editor)

        layout.addLayout(form_layout)

        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Parameters")
        refresh_btn.setFont(font)
        refresh_btn.clicked.connect(self.refresh_parameters)
        
        apply_btn = QPushButton("Apply Changes")
        apply_btn.setFont(font)
        apply_btn.clicked.connect(self.apply_parameters)
        
        btn_layout.addStretch()
        btn_layout.addWidget(refresh_btn)
        btn_layout.addWidget(apply_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _read_value(self, content, pattern, flags=0, group=1):
        match = re.search(pattern, content, flags)
        return match.group(group) if match else "Not Found"

    def _replace_value(self, content, key, new_val):
        # Generic regex replacer for standard OpenFOAM dictionary syntax
        return re.sub(rf"^(\s*{key}\s+)[-\d.eE+]+(;)", rf"\g<1>{new_val}\g<2>", content, flags=re.MULTILINE)

    def refresh_parameters(self):
        """Parses OpenFOAM files to populate the UI."""
        # 1. Read controlDict
        cd_path = os.path.join(self.case_dir, "system", "controlDict")
        if os.path.exists(cd_path):
            c = simple_read_file(cd_path)
            self.param_editors["endTime"].setText(self._read_value(c, r"^\s*endTime\s+([\d.eE+-]+);", re.MULTILINE))
            self.param_editors["deltaT"].setText(self._read_value(c, r"^\s*deltaT\s+([\d.eE+-]+);", re.MULTILINE))
            self.param_editors["writeInterval"].setText(self._read_value(c, r"^\s*writeInterval\s+([\d.eE+-]+);", re.MULTILINE))

        # 2. Read fvSolution
        fv_path = os.path.join(self.case_dir, "system", "fvSolution")
        if os.path.exists(fv_path):
            c = simple_read_file(fv_path)
            self.param_editors["p_relax"].setText(self._read_value(c, r"^\s*p\s+([\d.eE+-]+);", re.MULTILINE))
            self.param_editors["U_relax"].setText(self._read_value(c, r"^\s*U\s+([\d.eE+-]+);", re.MULTILINE))
            self.param_editors["k_relax"].setText(self._read_value(c, r"^\s*k\s+([\d.eE+-]+);", re.MULTILINE))
            self.param_editors["omega_relax"].setText(self._read_value(c, r"^\s*omega\s+([\d.eE+-]+);", re.MULTILINE))

        # 3. Read Rotation Speed (dynamicMeshDict or MRFProperties)
        dm_path = os.path.join(self.case_dir, "constant", "dynamicMeshDict")
        if os.path.exists(dm_path):
            c = simple_read_file(dm_path)
            def update_rpm(zone, key):
                # Match omega inside the zone block
                pat = re.compile(rf"{zone}\s*\{{.*?omega\s+([-\d.eE+-]+);", re.DOTALL)
                match = pat.search(c)
                if match:
                    try:
                        omega = float(match.group(1))
                        rpm = omega * 60 / (2 * math.pi) # Convert rad/s to RPM
                        self.param_editors[key].setText(f"{rpm:.0f}")
                    except ValueError:
                        self.log_message.emit(f"Error parsing omega for {zone}")
                else:
                    self.param_editors[key].setText("")
            
            update_rpm("rotation1", "rpm1")
            update_rpm("rotation2", "rpm2")
        else:
            # Fallback to MRFProperties
            self.log_message.emit("dynamicMeshDict missing, checking MRFProperties...")
            mrf_path = os.path.join(self.case_dir, "constant", "MRFProperties")
            if os.path.exists(mrf_path):
                c = simple_read_file(mrf_path)
                r1 = self._read_value(c, r"MRF1\s*\{.*?rpm\s+([-\d.eE+-]+);", re.DOTALL)
                if r1: self.param_editors["rpm1"].setText(r1)
                r2 = self._read_value(c, r"MRF2\s*\{.*?rpm\s+([-\d.eE+-]+);", re.DOTALL)
                if r2: self.param_editors["rpm2"].setText(r2)
            else:
                self.log_message.emit("Error: No rotation files found.")
                
        self.log_message.emit("Parameter refresh completed.")

    def apply_parameters(self):
        """Writes UI values back to OpenFOAM files."""
        self.log_message.emit("Applying changes...")
        try:
            # Update controlDict
            cd_path = os.path.join(self.case_dir, "system", "controlDict")
            c = simple_read_file(cd_path)
            c = self._replace_value(c, "endTime", self.param_editors["endTime"].text())
            c = self._replace_value(c, "deltaT", self.param_editors["deltaT"].text())
            c = self._replace_value(c, "writeInterval", self.param_editors["writeInterval"].text())
            simple_write_file(cd_path, c)

            # Update fvSolution
            fv_path = os.path.join(self.case_dir, "system", "fvSolution")
            c = simple_read_file(fv_path)
            c = self._replace_value(c, "p", self.param_editors["p_relax"].text())
            c = self._replace_value(c, "U", self.param_editors["U_relax"].text())
            c = self._replace_value(c, "k", self.param_editors["k_relax"].text())
            c = self._replace_value(c, "omega", self.param_editors["omega_relax"].text())
            simple_write_file(fv_path, c)

            # Update MRFProperties (regex is more specific here due to nested blocks)
            mrf_path = os.path.join(self.case_dir, "constant", "MRFProperties")
            if os.path.exists(mrf_path):
                c = simple_read_file(mrf_path)
                c = re.sub(r"(MRF1\s*\{[^}]*?rpm\s+)[-\d.eE+]+(;)", rf"\g<1>{self.param_editors['rpm1'].text()}\g<2>", c, flags=re.DOTALL)
                c = re.sub(r"(MRF2\s*\{[^}]*?rpm\s+)[-\d.eE+]+(;)", rf"\g<1>{self.param_editors['rpm2'].text()}\g<2>", c, flags=re.DOTALL)
                simple_write_file(mrf_path, c)
                self.log_message.emit("MRFProperties updated.")

            QMessageBox.information(self, "Success", "Parameters applied successfully.")

        except Exception as e:
            self.log_message.emit(f"[Error]: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply parameters:\n{e}")

# --- Main Window ---

class MainWindow(QMainWindow):
    confirmation_request_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepFlow Agent")
        self.setGeometry(100, 100, 1200, 800)
        
        # Init state variables
        self.confirmation_event_loop = None
        self.user_confirmation_response = None
        self.is_waiting_for_confirmation = False
        self.agent_executor = None
        self.chat_history = []
        self.current_instruction = ""
        self.solver_process = None
        self.current_attempt = 0
        self.max_attempts = 10
        self.last_agent_thought = "None"
        
        # Setup settings & logs
        self.settings = QSettings("DeepFlow", "AgentApp")
        self.load_app_settings()
        self.setup_logging()
        
        # Signals
        self.confirmation_request_signal.connect(self.on_confirmation_required)
        self.log_signal.connect(self.log)

        # Setup UI layout
        self.setup_ui()
        
        # Callbacks & Timers
        self.callback_handler = QtCallbackHandler(self)
        self.callback_handler.new_message.connect(self.log, Qt.QueuedConnection)
        
        self.plot_refresh_timer = QTimer(self)
        self.plot_refresh_timer.setInterval(10000)
        self.plot_refresh_timer.timeout.connect(self.process_and_refresh_plots)

        self.process_check_timer = QTimer(self)
        self.process_check_timer.setInterval(2000)
        self.process_check_timer.timeout.connect(self.check_solver_status)
        
        QTimer.singleShot(100, self.initialize_agent)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left Panel: Visualization & Parameters ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Visualization Area
        vis_frame = QFrame()
        vis_frame.setFrameShape(QFrame.StyledPanel)
        vis_layout = QVBoxLayout(vis_frame)
        vis_layout.addWidget(QLabel("<h2>Visualization Area</h2>", alignment=Qt.AlignCenter))
        
        self.fig_residuals = Figure()
        self.canvas_residuals = FigureCanvas(self.fig_residuals)
        self.ax_residuals = self.fig_residuals.add_subplot(111)
        
        self.fig_flow = Figure()
        self.canvas_flow = FigureCanvas(self.fig_flow)
        self.ax_flow = self.fig_flow.add_subplot(111)
        
        vis_layout.addWidget(self.canvas_residuals)
        vis_layout.addWidget(self.canvas_flow)
        
        self.refresh_plot_button = QPushButton("Refresh Plots")
        self.refresh_plot_button.clicked.connect(self.process_and_refresh_plots)
        vis_layout.addWidget(self.refresh_plot_button, alignment=Qt.AlignCenter)
        
        # Parameter Area
        params_frame = QFrame()
        params_frame.setFrameShape(QFrame.StyledPanel)
        params_layout = QVBoxLayout(params_frame)
        params_layout.addWidget(QLabel("<h2>Main Parameter Adjustment</h2>", alignment=Qt.AlignCenter))
        
        self.params_widget = ParameterWidget(self.case_dir)
        self.params_widget.log_message.connect(self.log_from_params)
        params_layout.addWidget(self.params_widget, 1)

        left_layout.addWidget(vis_frame, 2)
        left_layout.addWidget(params_frame, 1)
        splitter.addWidget(left_panel)

        # --- Right Panel: Chat & Controls ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        title = QLabel("<h2>DeepFlow Agent</h2>", alignment=Qt.AlignCenter)
        title.setFont(QFont(title.font().family(), 16, QFont.Bold))
        right_layout.addWidget(title)

        self.conversation_view = QTextEdit()
        self.conversation_view.setReadOnly(True)
        self.conversation_view.setFont(QFont("DejaVu Sans, Noto Color Emoji", 13))
        right_layout.addWidget(self.conversation_view)

        # Input Area
        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Please enter your instruction...")
        self.input_box.setFont(QFont(self.input_box.font().family(), 11))
        self.input_box.returnPressed.connect(self.handle_user_input)
        
        self.send_button = QPushButton("Send Instruction")
        self.send_button.setFont(QFont(self.send_button.font().family(), 11))
        self.send_button.clicked.connect(self.handle_user_input)
        self.send_button.setEnabled(False)
        
        input_layout.addWidget(self.input_box)
        input_layout.addWidget(self.send_button)
        right_layout.addLayout(input_layout)

        # Control Buttons
        btn_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Solver")
        self.run_button.clicked.connect(self.run_solver)
        
        self.kill_button = QPushButton("Kill Solver")
        self.kill_button.clicked.connect(self.kill_solver)
        self.kill_button.setEnabled(False)
        
        self.paraview_button = QPushButton("Launch Visualization")
        self.paraview_button.clicked.connect(self.launch_paraview)
        
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings_dialog)
        
        for btn in [self.run_button, self.kill_button, self.paraview_button, self.settings_button]:
            btn.setFont(QFont(btn.font().family(), 11))
            btn_layout.addWidget(btn)
        
        right_layout.addLayout(btn_layout)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])

    def setup_logging(self):
        """Creates log directory and file."""
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(os.path.dirname(base_path), 'log')
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_file_path = os.path.join(log_dir, f"DeepFlow_Conversation_{timestamp}.log")
            self.log(f"Logging to: {self.log_file_path}", 'system')
        except Exception as e:
            self.log(f"Log creation failed: {e}", 'error')
            self.log_file_path = None

    def load_app_settings(self):
        self.case_dir = self.settings.value("CASE_DIR", DEFAULT_SETTINGS["CASE_DIR"])
        self.solver_command = self.settings.value("SOLVER_COMMAND", DEFAULT_SETTINGS["SOLVER_COMMAND"])
        self.run_log_path = self.settings.value("RUN_LOG_PATH", DEFAULT_SETTINGS["RUN_LOG_PATH"])
        self.custom_api_base = self.settings.value("CUSTOM_API_BASE", DEFAULT_SETTINGS["CUSTOM_API_BASE"])
        self.model = self.settings.value("MODEL", DEFAULT_SETTINGS["MODEL"])
        
        # Set Env Vars
        env_keys = ["LANGCHAIN_TRACING_V2", "LANGCHAIN_ENDPOINT", "LANGCHAIN_API_KEY", "OPENAI_API_KEY"]
        for key in env_keys:
            os.environ[key] = self.settings.value(key, DEFAULT_SETTINGS[key])

    def open_settings_dialog(self):
        if SettingsDialog(self).exec_() == QDialog.Accepted:
            self.log("Settings saved. Restarting agent...", 'system')
            self.load_app_settings()
            self.send_button.setEnabled(False)
            QTimer.singleShot(100, self.initialize_agent)

    def initialize_agent(self):
        self.log("Initializing Agent...", 'system')
        try:
            if not os.environ.get('OPENAI_API_KEY'):
                raise ValueError("Missing OpenAI API Key.")

            self.agent_handler = OpenFOAMRAGAgent(case_dir=self.case_dir, 
                                                  openai_api_key=os.environ['OPENAI_API_KEY'], 
                                                  api_base=self.custom_api_base, 
                                                  model=self.model)
            
            # Wrap dangerous tools with confirmation logic
            final_tools = []
            confirm_map = {
                write_file.name: lambda **k: confirm_and_execute_wrapper(self, action_type='write', **k),
                search_and_replace_in_file.name: lambda **k: confirm_and_execute_wrapper(self, action_type='search_replace', **k),
                insert_line_into_file.name: lambda **k: confirm_and_execute_wrapper(self, action_type='insert', **k),
                delete_lines_from_file.name: lambda **k: confirm_and_execute_wrapper(self, action_type='delete', **k),
                execute_shell_command.name: lambda **k: confirm_and_execute_wrapper(self, action_type='execute', **k)
            }

            for tool in self.agent_handler.tools:
                if tool.name in confirm_map:
                    final_tools.append(StructuredTool.from_function(
                        func=confirm_map[tool.name],
                        name=tool.name,
                        description=tool.description,
                        args_schema=tool.args_schema
                    ))
                else:
                    final_tools.append(tool)

            self.agent_executor = AgentExecutor(agent=self.agent_handler.agent, tools=final_tools, 
                                                verbose=False, handle_parsing_errors=True)
            self.log(f"Agent Ready. Knowledge Base: {self.agent_handler.get_vectorstore_size()} items.", 'success')
            self.send_button.setEnabled(True)
        except Exception as e:
            self.log(f"Agent init failed: {e}", 'error')

    @pyqtSlot(str, str)
    def log(self, message: str, msg_type: str):
        """Appends message to UI and log file."""
        def update_ui():
            self.conversation_view.append(HtmlFormatter.format(message, msg_type))
            self.conversation_view.moveCursor(QTextCursor.End)
            self.conversation_view.ensureCursorVisible()
            # Scroll fix
            sb = self.conversation_view.verticalScrollBar()
            sb.setValue(sb.maximum())
        
        QTimer.singleShot(0, update_ui)
        
        if self.log_file_path:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] [{msg_type.upper()}]:\n{message}\n\n")

    @pyqtSlot(str)
    def log_from_params(self, message: str):
        msg_type = 'success' if "success" in message.lower() else 'error' if "error" in message.lower() else 'param'
        self.log(message, msg_type)

    def handle_user_input(self):
        if not self.agent_executor: return
        text = self.input_box.text().strip()
        if not text: return
        self.input_box.clear()
        
        self.log(text, 'user')
        
        if self.is_waiting_for_confirmation:
            self.handle_confirmation(text.lower())
        else:
            self.start_agent_interaction(text)

    def start_agent_interaction(self, text):
        self.initial_user_instruction = text
        self.current_instruction = text
        self.log("Task received. Processing...", 'system')
        self.send_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.current_attempt = 0
        self.run_agent_cycle()

    def handle_confirmation(self, text):
        self.user_confirmation_response = 'yes' if text in ['yes', 'y'] else 'no'
        if self.confirmation_event_loop and self.confirmation_event_loop.isRunning():
            self.confirmation_event_loop.quit()
        self.is_waiting_for_confirmation = False
        self.send_button.setEnabled(True)
        self.input_box.setFocus()

    def run_agent_cycle(self):
        self.current_attempt += 1
        if self.current_attempt > self.max_attempts:
            self.log("Max attempts reached.", 'error')
            self.send_button.setEnabled(True)
            return
        
        self.worker = AgentWorker(self.agent_executor, self.current_instruction, self.chat_history, self.callback_handler)
        self.worker.agent_finished.connect(self.on_agent_finished, Qt.QueuedConnection)
        self.worker.agent_error.connect(self.on_agent_error, Qt.QueuedConnection)
        self.worker.start()

    @pyqtSlot(str)
    def on_confirmation_required(self, message: str):
        self.log(message, 'confirm')
        self.is_waiting_for_confirmation = True
        self.send_button.setEnabled(True)
        self.input_box.setFocus()

    @pyqtSlot(dict)
    def on_agent_finished(self, result):
        summary = result.get('output', 'No summary.')
        self.log(summary, 'agent_final')
        self.log("Task complete.", 'system')
        self.params_widget.refresh_parameters()
        self.send_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.chat_history.extend([HumanMessage(content=self.current_instruction), AIMessage(content=summary)])

    @pyqtSlot(str)
    def on_agent_error(self, error_msg):
        self.log(error_msg, 'error')
        self.chat_history.extend([HumanMessage(content=self.current_instruction), AIMessage(content=f"Error: {error_msg}")])
        self.current_instruction = (f"Previous step failed. Analyze error and retry.\nOriginal Task: {self.initial_user_instruction}\nError: {error_msg}")
        self.log("Retrying with error context...", 'system')
        self.run_agent_cycle()

    def run_solver(self):
        if self.solver_process and self.solver_process.poll() is None:
            self.log("Solver already running.", 'system')
            return
        
        self.log(f"Starting solver: {self.solver_command}", 'system')
        try:
            # Use setsid to allow killing the entire process group later
            self.solver_process = subprocess.Popen(self.solver_command, cwd=self.case_dir, shell=True, 
                                                   text=True, encoding='utf-8', preexec_fn=os.setsid)
            self.log("Solver started in background.", 'success')
            self.run_button.setEnabled(False)
            self.kill_button.setEnabled(True)
            self.plot_refresh_timer.start()
            self.process_check_timer.start()
        except Exception as e:
            self.log(f"Start failed: {e}", 'error')

    def kill_solver(self):
        if self.solver_process and self.solver_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.solver_process.pid), signal.SIGTERM)
                self.log("Solver terminated.", 'success')
            except Exception as e:
                self.log(f"Termination failed: {e}", 'error')
            
            self.plot_refresh_timer.stop()
            self.process_check_timer.stop()
            self.solver_process = None
            self.run_button.setEnabled(True)
            self.kill_button.setEnabled(False)
            self.process_and_refresh_plots()

    def check_solver_status(self):
        if self.solver_process and self.solver_process.poll() is not None:
            self.log(f"Solver finished. Exit code: {self.solver_process.returncode}", 'success')
            self.plot_refresh_timer.stop()
            self.process_check_timer.stop()
            self.solver_process = None
            self.run_button.setEnabled(True)
            self.kill_button.setEnabled(False)
            self.process_and_refresh_plots()

    def launch_paraview(self):
        self.log("Launching ParaView...", 'system')
        try:
            subprocess.Popen(['paraFoam', '-builtin'], cwd=self.case_dir)
        except Exception as e:
            self.log(f"ParaView launch failed: {e}", 'error')

    def process_and_refresh_plots(self):
        """Parses log file and updates Matplotlib charts."""
        log_path = Path(self.run_log_path)
        if not log_path.exists():
            return
            
        try:
            residuals, flow = log2csv.parse_log_file(log_path)
            res_csv = os.path.join(self.case_dir, 'run_residuals.csv')
            flow_csv = os.path.join(self.case_dir, 'run_flowrate.csv')
            
            log2csv.write_residuals_csv(residuals, res_csv)
            log2csv.write_flow_rate_csv(flow, flow_csv)

            plot_residuals(self.ax_residuals, res_csv)
            self.canvas_residuals.draw()
            plot_flowrate(self.ax_flow, flow_csv)
            self.canvas_flow.draw()
            
            self.fig_residuals.tight_layout()
            self.fig_flow.tight_layout()
            self.log("Plots refreshed.", 'ui')
        except Exception as e:
            self.log(f"Plot refresh error: {e}", 'error')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setOrganizationName("DeepFlow")
    app.setApplicationName("AgentApp")
    
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
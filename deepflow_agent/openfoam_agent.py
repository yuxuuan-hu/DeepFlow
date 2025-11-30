import os
import subprocess
import shlex
import pandas as pd
from collections import deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage

# --- File size limit ---
MAX_LOG_FILE_SIZE = 0.5 * 1024 * 1024

# --- Tool definitions ---
@tool
def send_email(recipient_email: str, subject: str, body: str) -> str:
    """
    Send an email notification to the user when OpenFOAM computation terminates unexpectedly or encounters a serious error.
    This tool should be called after analyzing the logs and confirming the presence of critical issues that cannot be automatically resolved.

    :param recipient_email: The recipient's email address.
    :param subject: Email subject.
    :param body: Email body, should contain error summary and key portions of the log file.
    :return: A message string confirming that the email has been sent or failed to send.
    """
    # --- Simulated output ---
    print("\n--- Email Sent ---")
    print(f"Recipient: {recipient_email}")
    print(f"Subject: {subject}")
    print(f"Body:\n{body}")
    print("---------------------\n")
    
    return (f"Email has been generated and sent to {recipient_email}.")


@tool
def read_file(file_path: str) -> str:
    """
    Read the entire content of a file at the specified path.
    Note: If the target is a .log file and is larger than MAX_LOG_FILE_SIZE, this tool will refuse to read and prompt to use the 'query_log_file' tool.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
            
        file_size = os.path.getsize(file_path)

        if file_path.lower().endswith('.log') and file_size > MAX_LOG_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            return (f"Error: Log file '{os.path.basename(file_path)}' is too large to be read directly ({size_mb:.2f} MB). "
                    "Do not use this tool. Instead, use the 'query_log_file' tool to search for specific keywords or patterns within the log.")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def query_log_file(file_path: str, query: str) -> str:
    """
    Search for lines containing a specific query string in the specified (large) log file.
    This tool is the preferred method for analyzing large log files because it doesn't load the entire file into memory.
    It returns the last 50 matching lines found to facilitate analysis.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"

        matching_lines = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if query in line:
                    matching_lines.append(line)
                    if len(matching_lines) > 50:
                        matching_lines.pop(0)
        
        if not matching_lines:
            return f"No lines containing '{query}' found in '{os.path.basename(file_path)}'."
        
        result = (f"Found {len(matching_lines)} lines containing '{query}' in '{os.path.basename(file_path)}' "
                  f"(showing the last {len(matching_lines)}):\n---\n")
        result += "".join(matching_lines)
        result += "\n---"
        return result

    except Exception as e:
        return f"An error occurred while querying the file: {e}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write the specified content to the target file. If the file exists, it will overwrite the original content."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"File '{file_path}' has been successfully written."
    except Exception as e:
        return f"Error occurred while writing file: {e}"

@tool
def list_files(directory: str) -> str:
    """Recursively list all files and subdirectories in the specified directory."""
    if not os.path.isdir(directory):
        return f"Error: Directory '{directory}' does not exist."
    try:
        tree_structure = []
        for root, _, files in os.walk(directory):
            level = root.replace(directory, '').count(os.sep)
            indent = ' ' * 4 * level
            tree_structure.append(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                tree_structure.append(f"{sub_indent}{f}")
        return "\n".join(tree_structure)
    except Exception as e:
        return f"Error occurred while listing files: {e}"

@tool
def search_and_replace_in_file(file_path: str, search_string: str, replace_string: str) -> str:
    """
    Find all occurrences of 'search_string' in the file and replace them with 'replace_string'.
    This tool is more flexible and robust than replacing by line number.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        new_content = content.replace(search_string, replace_string)
        
        if original_content == new_content:
            return f"Text to replace '{search_string}' not found in file '{file_path}'."

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        return f"File '{file_path}' has successfully replaced '{search_string}' with '{replace_string}'."
    except Exception as e:
        return f"Error occurred while searching and replacing in file: {e}"

@tool
def insert_line_into_file(file_path: str, line_number: int, content_to_insert: str) -> str:
    """
    Insert a new line of content at the specified line number in the file. Existing content will be shifted down.
    Line numbers start from 1.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if line_number < 1 or line_number > len(lines) + 1:
            return f"Error: Line number {line_number} is out of insertable range (1-{len(lines) + 1})."
        
        lines.insert(line_number - 1, content_to_insert + '\n')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        return f"Content has been successfully inserted at line {line_number} in file '{file_path}'."
    except Exception as e:
        return f"Error occurred while inserting line into file: {e}"

@tool
def delete_lines_from_file(file_path: str, start_line: int, end_line: int = None) -> str:
    """
    Delete specified line or line range from file.
    Line numbers start from 1. If 'end_line' is not provided, only 'start_line' will be deleted.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if end_line is None:
            end_line = start_line

        if start_line < 1 or end_line > len(lines) or start_line > end_line:
             return f"Error: Line number range ({start_line}-{end_line}) is invalid. Total lines in file: {len(lines)}."

        del lines[start_line - 1 : end_line]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        return f"Successfully deleted lines {start_line} to {end_line} from file '{file_path}'."
    except Exception as e:
        return f"Error occurred while deleting lines from file: {e}"

@tool
def execute_shell_command(command: str, working_directory: str = None) -> str:
    """
    Execute a shell command in the specified directory and return its output for agent analysis.
    For security reasons, this tool will not execute complex shell syntax containing pipes (|), redirections (>, <), or chained commands (&&, ||).
    Please use simple, standalone commands.

    :param command: The terminal command string to execute (e.g.: "ls -l").
    :param working_directory: (Optional) The working directory path for command execution. If None, execute in the current directory.
    :return: A string containing the command's standard output, standard error, and exit code.
    """
    if not command:
        return "Error: No command provided."

    if any(op in command for op in ['|', '>', '<', '&&', '||', ';']):
        return "Error: For security reasons, complex commands containing pipes, redirections, or chained commands are not supported."

    try:
        args = shlex.split(command)
        result = subprocess.run(
            args, capture_output=True, text=True, cwd=working_directory, check=False
        )
        output = ""
        if result.stdout:
            output += f"--- Standard Output (stdout) ---\n{result.stdout.strip()}\n"
        if result.stderr:
            output += f"--- Standard Error (stderr) ---\n{result.stderr.strip()}\n"
        if not output:
            output = "Command executed, but no output generated.\n"
        output += f"\n--- Exit Code: {result.returncode} ---"
        return output
    except FileNotFoundError:
        return f"Error: Command '{shlex.split(command)[0]}' not found. Please ensure it is installed and in the system PATH."
    except Exception as e:
        return f"Unknown error occurred while executing command: {e}"

@tool
def read_csv_file(file_path: str) -> str:
    """
    Read a CSV or .dat file at the specified path and return its last 50 lines of content.
    Commonly used for reading OpenFOAM residual files and other monitoring data.
    """
    if not file_path.lower().endswith(('.csv', '.dat')):
         return f"Error: File '{os.path.basename(file_path)}' is not a supported CSV or .dat file."
    try:
        if not os.path.exists(file_path):
            return f"Error: File does not exist at {file_path}"
        df = pd.read_csv(file_path, comment='#', delim_whitespace=True, header=None)
        
        if df.empty:
            return f"File '{os.path.basename(file_path)}' is empty or contains only comments."

        num_columns = len(df.columns)
        if num_columns > 0:
            df.columns = [f'column_{i+1}' for i in range(num_columns)]

        tail_data = df.tail(50)
        
        result = (f"Last {len(tail_data)} lines of content from file '{os.path.basename(file_path)}':\n---\n")
        result += tail_data.to_string(index=False)
        result += "\n---"
        return result
    except Exception as e:
        return f"Error occurred while reading CSV/DAT file: {e}"

@tool
def read_log_tail(file_path: str) -> str:
    """
    Read the last 50 lines of content from a log file at the specified path.
    This is the preferred tool for monitoring real-time progress after starting computation.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File does not exist at {file_path}"

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            last_lines = deque(f, 50)
        
        if not last_lines:
            return f"File '{os.path.basename(file_path)}' is empty."
            
        result = (f"Last {len(last_lines)} lines of content from file '{os.path.basename(file_path)}':\n---\n")
        result += "".join(last_lines)
        result += "---"
        return result
    except Exception as e:
        return f"Error occurred while reading log file tail: {e}"


class OpenFOAMRAGAgent:
    def __init__(self, case_dir: str, openai_api_key: str, api_base: str, model: str):
        if not os.path.isdir(case_dir):
            raise FileNotFoundError(f"Case directory '{case_dir}' does not exist.")
        self.case_dir = case_dir
        self.tools = [
            read_file, write_file, list_files, search_and_replace_in_file,
            insert_line_into_file, delete_lines_from_file, execute_shell_command,
            read_csv_file, read_log_tail, send_email
        ]
        self.llm = ChatOpenAI(model=model, temperature=0, openai_api_key=openai_api_key, openai_api_base=api_base)
        self.chat_history = []
        self._setup_rag(api_base)
        self.agent = self._create_agent()
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def _setup_rag(self, api_base: str):
        print("\n--- RAG: Starting to load knowledge base... ---")
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_base=api_base)
            db_index_path = "faiss_index/faiss_index_database"
            tutorials_index_path = "faiss_index/faiss_index_tutorials"
            
            if not os.path.exists(db_index_path) or not os.path.exists(tutorials_index_path):
                print("\n!!! RAG Critical Error !!!")
                print(f"One or more pre-built index folders ('{db_index_path}', '{tutorials_index_path}') do not exist.")
                print("Please run 'python scripts/create_knowledge_base_sync.py' or 'python scripts/create_knowledge_base_async.py' script first to generate them.")
                print("RAG functionality will be unavailable.\n")
                return

            print("--- RAG: Loading pre-built static knowledge base index... ---")
            db_vs = FAISS.load_local(db_index_path, embeddings, allow_dangerous_deserialization=True)
            vectorstore = db_vs
            print(f"- Loaded database index, containing {vectorstore.index.ntotal} vectors.")

            tutorials_vs = FAISS.load_local(tutorials_index_path, embeddings, allow_dangerous_deserialization=True)
            vectorstore.merge_from(tutorials_vs)

            print(f"--- RAG: Knowledge base ready, finally containing {vectorstore.index.ntotal} vectors. ---\n")
            self.vectorstore_size = vectorstore.index.ntotal
            retriever = vectorstore.as_retriever()
            
            retriever_tool = create_retriever_tool(
                retriever, 
                "openfoam_knowledge_retriever", 
                "Search and retrieve content from OpenFOAM official tutorials and local case database. Use this tool when you need to understand official tutorial examples or other case configurations."
            )
            self.tools.append(retriever_tool)
            
        except Exception as e:
            print(f"--- RAG Error: An error occurred while building RAG retriever: {e}. Will continue with basic file tools only. ---\n")

    def _create_agent(self):
        """Create OpenAI Tools agent."""
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are a world-class OpenFOAM expert, an AI engineer capable of autonomously operating files.
                Your task is to solve problems by directly calling tools to modify files based on user instructions and subsequent error feedback.
                Current case absolute path is: {self.case_dir}, user's email address is example@deepflow.com.

                **Core Workflow and Rules:**

                1.  **Analysis and Exploration**: First, deeply understand user requirements or error logs. If needed, use `list_files` and `read_file` tools to check file structure and content, or use `openfoam_knowledge_retriever` to retrieve background knowledge.

                2.  **Thinking**: Before deciding to call any file modification tool (such as `write_file`, `search_and_replace_in_file`, etc.) or execute commands (`execute_shell_command`), you **must** first state your **"intention"** or **"reasoning"** in one or two concise natural language sentences.
                    -   **Good Example**: "To improve computational stability, I need to change the discretization scheme for U in fvSchemes from 'linearUpwind' to 'linearUpwind limited'."

                3.  **Action**: After stating your "reasoning", **immediately** call the corresponding tool to perform the operation.
                    -   **Prefer Precise Modifications**: Prioritize using `search_and_replace_in_file`, `insert_line_into_file`, `delete_lines_from_file` for precise and safe file modifications.
                    -   **Fallback to Full File Overwrite**: Only use `write_file` when performing extensive complex modifications that cannot be achieved with precise tools.
                    -   **Execute Commands**: Use `execute_shell_command` to run scripts or check system status.

                4.  **Monitoring and Reporting**: After using `execute_shell_command` to start a long-running computational task (such as 'blockMesh' or 'simpleFoam'), you should **check the task status only once** and report to the user. Use `read_log_tail` to read log files, or use `read_csv_file` to read residual files, then summarize the current status to the user (e.g., "Computation is proceeding normally, residuals have decreased to X" or "Mesh generation successful"), then end your turn. **Do not** continuously monitor.

                5.  **Error Handling and Notification**: If the return result from `execute_shell_command` has a non-zero `exit code`, this indicates command execution failure. You must:
                    a.  Carefully analyze the `standard error (stderr)` content to determine the cause of failure.
                    b.  If this is an expected, solvable configuration issue, try to fix it as before.
                    c.  If this is an unexpected, serious computational crash (e.g., solver divergence), you **must** call the `send_email` tool. Clearly describe the problem in the email body and attach the last few lines of the log file, then report to the user that you have sent an email notification.
                    d.  **As long as computation is proceeding normally (exit code is 0), never proactively suggest modifying parameters.**

                6.  **Path Rules**: When calling any file operation tool, you **must** use absolute paths starting with `{self.case_dir}`.

                7.  **Output Format**: All your outputs must be plain text. **Strictly prohibited** from using Markdown (`#`, `*`, ````) or any other formatting syntax.
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        return create_openai_tools_agent(self.llm, self.tools, prompt)

    def get_vectorstore_size(self):
        return getattr(self, 'vectorstore_size', 0)

    def run(self, user_input: str):
        """
        Run agent and manage conversation history.
        """
        result = self.executor.invoke({
            "input": user_input,
            "chat_history": self.chat_history 
        })
        
        output = result.get("output", "Failed to retrieve output.")
        
        self.chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=output),
        ])
        
        self.chat_history = self.chat_history[-20:]
        
        return output
# DeepFlow - Intelligent Computational Fluid Dynamics Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-v2412-green.svg)](https://www.openfoam.com/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-orange.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-yellow.svg)](https://python.langchain.com/)

## 📖 Project Overview

DeepFlow is an intelligent Computational Fluid Dynamics (CFD) assistant system that adopts the "Think-Act" paradigm, integrating OpenFOAM solvers, RAG (Retrieval-Augmented Generation) technology, and a modern graphical user interface. The system aims to break down the high technical barriers in traditional CFD simulations, helping engineers and researchers perform simulation analysis, parameter tuning, result visualization, and problem diagnosis more efficiently through natural language interaction.

🏆 This project won the **Second Prize in the Professional Track of the 3rd National College Student Intelligent Terminal Simulation Technology Competition**.

> ⚠️ **Open Source Notice**: Due to the large size of the complete CFD case files (including mesh and transient data), this repository currently **only open-sources the core code and architecture implementation of the Agent component**.

<div align="center">
  <img src="figs/overall.png" width="800" alt="DeepFlow Agent Architecture" />
</div>

## ✨ Key Features

* 🤖 **Intelligent AI Assistant**: An intelligent dialogue system built on LangChain and large language models, supporting natural language-driven control of the entire simulation workflow.
* 🔍 **RAG Retrieval System**: Integrates FAISS vector database with professional knowledge bases to ensure the professionalism and accuracy of AI responses.
* 🖥️ **Modern GUI**: An intuitive graphical interface developed with PyQt5, integrating parameter configuration, solver control, and 3D visualization windows.
* 📊 **Real-time Monitoring and Post-processing**: Integrates Matplotlib and PyVista, supporting real-time tracking of residual curves and flow rate monitoring.

## 🔄 Agent Workflow

DeepFlow Agent adopts a decision-making loop based on the ReAct framework, transforming complex CFD simulation processes into automated tasks of "Perception-Planning-Execution".

<div align="center">
  <img src="figs/workflow.png" width="800" alt="Agent Workflow" />
</div>

1.  **Intent Recognition**: Users input simulation requirements through natural language (e.g., "Run transient calculation and monitor outlet flow rate").
2.  **Knowledge Retrieval (RAG)**: The Agent automatically retrieves OpenFOAM configuration rules and case libraries to assist in modifying configuration files.
3.  **Tool Invocation**: The agent automatically executes operations such as file read/write, mesh conversion, and solver startup through a toolset, and parses log files in real-time to provide simulation status feedback.

## 💻 App Interface and Interaction

The system interface design follows engineers' operational habits, deeply integrating AI dialogue with professional simulation control panels.

<div align="center">
  <img src="figs/UI_layout.png" width="800" alt="App Interface" />
</div>

* **Dialogue Interaction Area**: Users send commands here, and AI provides real-time feedback on operation results and suggestions.
* **Visualization Panel**: Supports quick viewing of mesh previews and flow field contour maps.
* **Real-time Monitoring**: Dynamically plots residual and flow curves to help users assess convergence.

## 🧪 Simulation Results Showcase

This project uses a dual-fan cooling system for laptop computers as a case study.

### 1. Flow Field Characteristics Analysis
The system successfully captured the complex flow characteristics in the dual-fan counter-rotating system. Steady-state simulations showed a maximum flow velocity of **11.77 m/s**, and transient results clearly demonstrated periodic pulsation and vortex shedding phenomena around the blades.

<div align="center">

| Steady-State Velocity Field | Transient Velocity Field |
| :---: | :---: |
| <img src="figs/steady_velocity.jpg" width="400" /> | <img src="figs/transient_velocity.jpg" width="400" /> |

</div>


### 2. Streamlines and Topological Structure
Through streamline tracking, the spiral motion trajectory of airflow under the suction of the impeller and the flow straightening effect at the outlet are clearly presented.

<div align="center">
  <img src="figs/streamline.png" width="400" alt="Flow Streamlines" />
</div>

### 3. Local Flow Field Structure
Through pressure-colored cross-sectional streamline plots (LIC), we can clearly identify key flow features inside the fan housing.
* **Strong Suction Zone**: Streamlines near the inlet show significant inward bending, reflecting the powerful suction mechanism of the impeller.
* **Vortices and Recirculation**: The diagram clearly captures complex vortex structures and recirculation zones near the housing. These local unstable flow phenomena are considered the main sources of aerodynamic noise in the system.

<div align="center">
  <img src="figs/surface_LIC.png" width="800" alt="Surface LIC Analysis" />
</div>

## 🏗️ Project Structure

```

DeepFlow/
├── deepflow_agent/          # Core modules
│   ├── main_gui.py          # Main graphical interface
│   ├── openfoam_agent.py    # OpenFOAM operations and intelligent tool integration
│   ├── plot_data.py         # Data visualization and plotting
│   └── log2csv.py           # Log conversion tool
├── dataset/                 # Knowledge base examples
│   └── deepflow_agent.jsonl
├── cavity/                  # Cavity flow sample case
├── mixerVessel2D/           # Mixer 2D MRF sample case
├── figs/  
├── scripts/                 # Auxiliary scripts (build knowledge base index, word cloud, etc.)
│   ├── create_knowledge_base_sync.py
│   ├── create_knowledge_base_async.py
│   └── create_wordcloud.py
├── requirements.txt  
├── .gitignore  
└── README.md

```

## 🛠️ Configuration

### Environment Configuration

Modify the following configuration in `deepflow_agent/main_gui.py`:

```python
DEFAULT_SETTINGS = {
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_ENDPOINT": "[https://api.smith.langchain.com](https://api.smith.langchain.com)",
    "LANGCHAIN_API_KEY": "YOUR_LANGCHAIN_API_KEY",
    "OPENAI_API_KEY": "YOUR_OPENAI_API_KEY",
    "CUSTOM_API_BASE": "YOUR_CUSTOM_API_BASE",
    "CASE_DIR": "YOUR_CASE_DIRECTORY_PATH",
    # Replace 'icoFoam' with your specific solver, keep the redirection tail
    "SOLVER_COMMAND": "icoFoam > run.log 2>&1 &",
    "RUN_LOG_PATH": "YOUR_CASE_DIRECTORY_PATH/run.log",
    "MODEL": "qwen3-coder-plus"
}
```

### Knowledge Base Configuration

Configure in `database/create_knowledge_base_sync.py` or `database/create_knowledge_base_async.py`:

```python
TUTORIALS_DIR = "/path/to/your/tutorials"
DATABASE_DIR = "/path/to/your/database"
```

## 🚀 Quick Start

### System Requirements

- **Operating System**: Linux
- **Python**: 3.10
- **OpenFOAM**: v2412


### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yuxuuan-hu/DeepFlow.git
   cd DeepFlow
   ```
2.  **Create Conda Environment**

    ```bash
    conda create -n deepflow python=3.10 -y
    pip install -r requirements.txt
    conda activate deepflow
    ```

3.  **Build Knowledge Base**

    ```bash
    cd scripts
    # Run the synchronous builder (recommended for smaller datasets)
    python create_knowledge_base_sync.py

    # Or, run the asynchronous builder (useful for larger datasets / batch indexing)
    python create_knowledge_base_async.py
    ```

### Run the Application

```bash
cd deepflow_agent
python main_gui.py
```



## 📚 User Guide

### Basic Operation Workflow

1.  **Launch Application**: Run the main program and wait for the GUI interface to load
2.  **Start Conversation**: Interact with the AI assistant in the chat interface
3.  **Case Preparation**: The assistant retrieves from the knowledge base based on RAG to help check or modify configuration files
4.  **Execute Simulation**: Call the solver to start computation
5.  **Result Analysis**: View real-time charts and simulation results

## 🔧 Development Guide

### Adding New Tools

Add new tool functions in `openfoam_agent.py`:

```python
@tool
def your_new_tool(param: str) -> str:
    """Tool description"""
    # Implementation logic
    return "result"
```

### Extending Knowledge Base

1.  Add new documents in the `database/` directory
2.  Run `create_knowledge_base_sync.py` or `create_knowledge_base_async.py` to rebuild the index

### Customizing the Interface

Modify GUI components and layout in `main_gui.py`.

## 📊 Test Case Descriptions

### Mixer Vessel 2D MRF (mixerVessel2DMRF)

  - **Type**: 2D Multiple Reference Frame simulation
  - **Solver**: pimpleFoam
  - **Features**: Coupling of rotating mixer and stationary vessel

### Cavity Flow (cavity)

  - **Type**: Classic validation case
  - **Solver**: icoFoam
  - **Features**: Lid-driven cavity flow

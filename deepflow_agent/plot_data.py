# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
import sys

def custom_log_formatter(y, pos):
    """
    Format Y-axis tick values as '10^n' for log scale.
    """
    try:
        if y <= 0: return ""
        exponent = np.floor(np.log10(y))
        if np.isclose(y, 10**exponent):
            return f"$10^{{{int(exponent)}}}$"
        else:
            return ""
    except (ValueError, TypeError):
        return ""

def plot_residuals(ax, csv_path):
    """
    Plot residuals data from CSV on the given Axes.
    """
    ax.clear()
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"Error: File not found\n{csv_path}", 
                ha='center', va='center', color='red')
        return
    except Exception as e:
        ax.text(0.5, 0.5, f"Error reading file:\n{e}", 
                ha='center', va='center', color='red')
        return

    font_prop = fm.FontProperties()

    min_time = df['Time'].min() if 'Time' in df.columns else 0

    for column in df.columns:
        if column != 'Time':
            ax.plot(df['Time'], df[column].abs(), label=column, marker='o',
                    linestyle='-', markersize=0.25, linewidth=0.5)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(custom_log_formatter))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, pos: ""))
    ax.set_title('Residual Changes', fontproperties=font_prop, fontsize=18)
    ax.set_xlabel('Time (s)', fontproperties=font_prop, fontsize=16)
    ax.set_ylabel('Final Residual (log scale)', fontproperties=font_prop, fontsize=16)
    ax.set_xlim(left=min_time)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(prop=font_prop, fontsize=25)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(16)
    ax.grid(True, which="both", ls="--", linewidth=0.5)

def plot_flowrate(ax, csv_path):
    """
    Plot flow rate data from CSV on the given Axes.
    """
    ax.clear()
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"Error: File not found\n{csv_path}", 
                ha='center', va='center', color='red')
        return
    except Exception as e:
        ax.text(0.5, 0.5, f"Error reading file:\n{e}", 
                ha='center', va='center', color='red')
        return

    font_prop = fm.FontProperties()

    label_map = {
        'flowRateInlet': 'Inlet Flow Rate 1',
        'flowRateInlet1': 'Inlet Flow Rate 2',
        'flowRateOutlet': 'Outlet Flow Rate 1',
        'flowRateOutlet1': 'Outlet Flow Rate 2',
    }

    min_time = df['Time'].min() if 'Time' in df.columns else 0

    for column in df.columns:
        if column != 'Time':
            label = label_map.get(column, column)
            ax.plot(df['Time'], df[column], label=label, marker='o',
                    linestyle='-', markersize=0.25, linewidth=0.5)

    ax.set_title('Flow Rate Changes', fontproperties=font_prop, fontsize=18)
    ax.set_xlabel('Time (s)', fontproperties=font_prop, fontsize=16)
    ax.set_ylabel('Flow Rate (mm³/s)', fontproperties=font_prop, fontsize=16)
    ax.set_xlim(left=min_time)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(prop=font_prop, fontsize=25)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(16)
    ax.grid(True, which="both", ls="--", linewidth=0.5)

if __name__ == '__main__':
    print("--- Running in standalone test mode ---")
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.default'] = 'regular'

    test_residuals_csv_file = 'YOUR_CASE_DIRECTORY_PATH/run_residuals.csv'
    test_flowrate_csv_file = 'YOUR_CASE_DIRECTORY_PATH/run_flowrate.csv'

    from pathlib import Path
    if not Path(test_residuals_csv_file).is_file():
        print(f"Error: Test file does not exist at path '{test_residuals_csv_file}'", file=sys.stderr)
        print("Please verify the path is correct, or modify the test_csv_file variable in the script.", file=sys.stderr)
        sys.exit(1)

    fig, (residual_ax, flowrate_ax) = plt.subplots(2, 1, figsize=(8, 8))
    plot_residuals(residual_ax, test_residuals_csv_file)
    plot_flowrate(flowrate_ax, test_flowrate_csv_file)
    fig.tight_layout()
    print("Displaying test charts...")
    plt.show()
    print("--- Test completed ---")

    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    residuals_img_path = os.path.join(tmp_dir, "residuals.png")
    flowrate_img_path = os.path.join(tmp_dir, "flowrate.png")

    from matplotlib.figure import Figure

    fig_residual = Figure(figsize=(8, 4))
    ax_residual = fig_residual.add_subplot(1, 1, 1)
    plot_residuals(ax_residual, test_residuals_csv_file)
    fig_residual.tight_layout()
    fig_residual.savefig(residuals_img_path, bbox_inches='tight', dpi=200)
    print(f"Residuals plot saved separately to: {residuals_img_path}")

    fig_flowrate = Figure(figsize=(8, 4))
    ax_flowrate = fig_flowrate.add_subplot(1, 1, 1)
    plot_flowrate(ax_flowrate, test_flowrate_csv_file)
    fig_flowrate.tight_layout()
    fig_flowrate.savefig(flowrate_img_path, bbox_inches='tight', dpi=200)
    print(f"Flow rate plot saved separately to: {flowrate_img_path}")

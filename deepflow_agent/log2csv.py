# -*- coding: utf-8 -*-

import re
import csv
import argparse
from pathlib import Path
import sys

def parse_log_file(log_path):
    """
    Parse OpenFOAM log file and extract solver residuals and flow rate values for each time step.

    Args:
        log_path (Path): Path object to the log file.

    Returns:
        tuple: A tuple containing two lists (residuals_data, flow_rate_data).
               residuals_data: List containing residual data for all time steps.
               flow_rate_data: List containing flow rate data for all time steps.
    """
    # --- Regular expression definitions ---
    # **This is the key fix**
    time_pattern = re.compile(r'^Time = ([\d\.eE\-+]+)s?$') # Fixed support for scientific notation
    solver_pattern = re.compile(
        r'^(?:smoothSolver|GAMG|DICPCG):\s+Solving for (\w+),.*Final residual = ([\d\.\-eE+]+)' # Added DICPCG
    )
    continuity_pattern = re.compile(
        r'^time step continuity errors.*cumulative = ([\d\.\-eE+]+)$'
    )
    flow_rate_name_pattern = re.compile(r'^\s*surfaceFieldValue (\w+) write:')
    flow_rate_value_pattern = re.compile(r'^\s*sum\(.+\) of phi = ([\d\.\-eE+]+)')

    # --- Data storage initialization ---
    all_residuals_data = []
    all_flow_rate_data = []
    current_residuals = {}
    current_flow_rates = {}
    last_flow_rate_name = None

    print(f"📄 Reading log file: {log_path}")
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 1. Check if this is a new time step
                time_match = time_pattern.match(line)
                if time_match:
                    # --- Save data from previous time step ---
                    if current_residuals and 'Time' in current_residuals:
                        all_residuals_data.append(current_residuals)
                    
                    if current_flow_rates and len(current_flow_rates) > 1:
                        all_flow_rate_data.append(current_flow_rates)

                    # Create data dictionary for new time step
                    time_value = time_match.group(1)
                    current_residuals = {'Time': time_value}
                    current_flow_rates = {'Time': time_value}
                    last_flow_rate_name = None
                    continue

                if 'Time' not in current_residuals:
                    continue

                # 2. Check solver residuals
                solver_match = solver_pattern.match(line)
                if solver_match:
                    variable = solver_match.group(1)
                    final_residual = solver_match.group(2)
                    # pcorr is the correction for p, generally we care about p's residual
                    if variable in ['Ux', 'Uy', 'Uz', 'p', 'omega', 'k']:
                        current_residuals[variable] = final_residual
                    continue

                # 3. Check continuity errors
                continuity_match = continuity_pattern.search(line)
                if continuity_match:
                    cumulative_error = continuity_match.group(1)
                    current_residuals['continuity'] = cumulative_error
                    continue

                # 4. Check flow rate name line
                flow_name_match = flow_rate_name_pattern.match(line)
                if flow_name_match:
                    last_flow_rate_name = flow_name_match.group(1)
                    continue

                # 5. Check flow rate value line
                flow_value_match = flow_rate_value_pattern.match(line)
                if flow_value_match and last_flow_rate_name:
                    flow_value = flow_value_match.group(1)
                    current_flow_rates[last_flow_rate_name] = flow_value
                    last_flow_rate_name = None
                    continue

        # --- After loop ends, add data from the last time step ---
        if current_residuals and 'Time' in current_residuals:
            all_residuals_data.append(current_residuals)
        
        if current_flow_rates and len(current_flow_rates) > 1:
            all_flow_rate_data.append(current_flow_rates)

    except Exception as e:
        print(f"❌ Error reading file: {e}", file=sys.stderr)
        return [], []

    print(f"📊 Extracted residual data for {len(all_residuals_data)} time steps.")
    print(f"💧 Extracted flow rate data for {len(all_flow_rate_data)} time steps.")
    return all_residuals_data, all_flow_rate_data
def write_residuals_csv(data, output_path):
    """
    Write extracted residual data to CSV file.
    """
    if not data:
        print("⚠️ No residual data extracted, skipping residuals CSV file generation.")
        return

    headers = ['Time', 'Ux', 'Uy', 'Uz', 'p', 'continuity', 'omega', 'k']
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        print(f"✅ Successfully created residuals CSV file: {output_path}")
    except IOError as e:
        print(f"❌ Error writing residuals CSV file: {e}", file=sys.stderr)

def write_flow_rate_csv(data, output_path):
    """
    Write extracted flow rate data to CSV file.
    """
    if not data:
        print("⚠️ No flow rate data extracted, skipping flowrate CSV file generation.")
        return

    header_set = set()
    for row in data:
        header_set.update(row.keys())
    
    if 'Time' in header_set:
        header_set.remove('Time')
        headers = ['Time'] + sorted(list(header_set))
    else:
        headers = sorted(list(header_set))

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        print(f"✅ Successfully created flow rate CSV file: {output_path}")
    except IOError as e:
        print(f"❌ Error writing flow rate CSV file: {e}", file=sys.stderr)

def main():
    """
    Main script function, handles command-line arguments and calls core functionality.
    """
    parser = argparse.ArgumentParser(
        description="Convert residual and flow rate data from OpenFOAM log files into two separate CSV files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'logfile',
        type=str,
        help="Path to the log file to be processed."
    )
    args = parser.parse_args()

    log_file_path = Path(args.logfile)

    if not log_file_path.is_file():
        print(f"❌ Error: File '{log_file_path}' does not exist or is not a valid file.", file=sys.stderr)
        sys.exit(1)

    base_name = log_file_path.stem
    output_dir = log_file_path.parent
    residuals_csv_path = output_dir / f"{base_name}_residuals.csv"
    flow_rate_csv_path = output_dir / f"{base_name}_flowrate.csv"

    residuals_data, flow_rate_data = parse_log_file(log_file_path)
    
    write_residuals_csv(residuals_data, residuals_csv_path)
    write_flow_rate_csv(flow_rate_data, flow_rate_csv_path)

if __name__ == "__main__":
    main()
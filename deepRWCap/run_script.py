#!/usr/bin/env python3
import subprocess
import re
import sys
import numpy as np
import os
from datetime import datetime

TESTCASE_FOLDER = "/workspace/testcases/cap3d"
REFERENCE_FOLDER = "/workspace/testcases/cap3d/ref"
TESTCASE_SETTING = {
    'case1' : '-p 0.01 -c 0.01 --c-ratio 0.95',
    'case2' : '-p 0.01 -c 0.01 --c-ratio 0.95',
    'case3' : '-p 0.01 -c 0.01 --c-ratio 0.95',
    'case4' : '-p 0.01 -c 0.01 --c-ratio 0.95',
    'case5' : '-p 0.01 -c 0.01 --c-ratio 0.95',
    'case6' : '-p 0.01 -c 0.01 --c-ratio 0.95',
    'case7' : '-p 0.01 -c 0.01 --c-ratio 0.3',
    'case8' : '-p 0.01 -c 0.01 --c-ratio 0.3',
    'case9' : '-p 0.01 -c 0.01 --c-ratio 0.3',
    'case10': '-p 0.01 -c 0.01 --c-ratio 0.3'
}

def parse_cap_unit(s: str):
    s = s.strip()
    if s.endswith("pF"):
        s = float(s[:-2]) * 1e-12
    elif s.endswith("fF"):
        s = float(s[:-2]) * 1e-15
    return float(s)

def parse_spice(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    capmat = {}
    for line in lines:
        if line.startswith("** ") and line.count("(total)"):
            tokens = line.split(" ")
            capmat[tokens[1]] = {} 
            capmat[tokens[1]][tokens[1]] = parse_cap_unit(tokens[2])
    for line in lines:
        if line.startswith("Cp"):
            tokens = line.split(" ")
            if tokens[2] == "0":
                tokens[2] = "GROUND"
            coupling_cap = parse_cap_unit(tokens[3])
            if tokens[1] in capmat and tokens[2] in capmat[tokens[1]]:
                coupling_cap = (capmat[tokens[1]][tokens[2]] + coupling_cap) / 2
            capmat[tokens[1]][tokens[2]] = coupling_cap
            if tokens[2] in capmat:
                capmat[tokens[2]][tokens[1]] = coupling_cap
    return capmat

def parse_dspf(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    capmat = {}
    for line in lines:
        if line.startswith("*|NET "):
            tokens = line.split(" ")
            capmat[tokens[1]] = {} 
            capmat[tokens[1]][tokens[1]] = float(tokens[2])
    for line in lines:
        if line.startswith("C"):
            tokens = line.split(" ")
            if tokens[2] == "0":
                tokens[2] = "GROUND"
            coupling_cap = float(tokens[3])
            if tokens[1] in capmat and tokens[2] in capmat[tokens[1]]:
                coupling_cap = (capmat[tokens[1]][tokens[2]] + coupling_cap) / 2
            capmat[tokens[1]][tokens[2]] = coupling_cap
            if tokens[2] in capmat: 
                capmat[tokens[2]][tokens[1]] = coupling_cap
    return capmat

def parse_rwcap_out(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    
    capmat = {}
    current_master = None
    num_walks = None
    hops_per_walk = None
    
    for line in lines:
        if line.startswith("Master "):
            tokens = line.split(" ")
            current_master = tokens[1]
            capmat[current_master] = {}
        elif line.startswith("Capacitance on "):
            tokens = line.split(" ")
            capmat[current_master][tokens[2]] = abs(float(tokens[4]))
        elif line.startswith("RWCap has run "):
            # Parse: "RWCap has run 88576 walks (8.13 hops/walk)"
            tokens = line.split()
            num_walks = int(tokens[3])
            # Extract hops/walk from the parentheses
            if "hops/walk)" in line:
                hops_match = re.search(r'\(([\d.]+)\s*hops/walk\)', line)
                if hops_match:
                    hops_per_walk = float(hops_match.group(1))
    
    return capmat, num_walks, hops_per_walk

def parse_gpu_stats_from_log(file_name):
    """Parse GPU sampler statistics from the log file"""
    if not os.path.exists(file_name):
        return None
    
    gpu_stats = {
        'green_total_tasks': None,
        'green_avg_batch_size': None,
        'gradient_total_tasks': None,
        'gradient_avg_batch_size': None
    }
    
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    in_gpu_section = False
    current_sampler = None
    
    for line in lines:
        line = line.strip()
        
        if line == "=== GPU Sampler Statistics ===":
            in_gpu_section = True
            continue
            
        if in_gpu_section:
            if "Green's Function Sampler:" in line:
                current_sampler = 'green'
            elif "Gradient Sampler:" in line:
                current_sampler = 'gradient'
            elif "Total tasks processed:" in line:
                # Split by colon and get the number
                parts = line.split(':')
                if len(parts) >= 2:
                    tasks_str = parts[1].strip()
                    tasks = int(tasks_str)
                    if current_sampler == 'green':
                        gpu_stats['green_total_tasks'] = tasks
                    elif current_sampler == 'gradient':
                        gpu_stats['gradient_total_tasks'] = tasks
            elif "Average batch size:" in line:
                # Split by colon, then take the number
                parts = line.split(':')
                avg_size = float(parts[1])
                if current_sampler == 'green':
                    gpu_stats['green_avg_batch_size'] = avg_size
                elif current_sampler == 'gradient':
                    gpu_stats['gradient_avg_batch_size'] = avg_size

            elif "Task" in line and "random walk costed" in line:
                # End of GPU stats section
                in_gpu_section = False
                break
    
    return gpu_stats

def parse_result(file_name):
    if not os.path.exists(file_name):
        return None, None, None
    capmat = None
    num_walks = None
    hops_per_walk = None
    if file_name.endswith(".spice"):
        capmat = parse_spice(file_name)
    elif file_name.endswith(".dspf"):
        capmat = parse_dspf(file_name)
    else:
        capmat, num_walks, hops_per_walk = parse_rwcap_out(file_name)
    print(f"Parsed {file_name}, found {len(capmat)} masters, {sum(len(v) for v in capmat.values())} capacitances.")
    return capmat, num_walks, hops_per_walk

def compute_error(mat, ref_mat):
    if mat is None or ref_mat is None:
        return float('NaN'), {}
    abserr = 0
    abssum = 0
    selfcap_err = {}
    for master_name, row in ref_mat.items():
        if master_name not in mat or master_name not in mat[master_name]:
            continue
        selfcap_err[master_name] = abs(mat[master_name][master_name] / row[master_name] - 1)
        for coupling_name, ref_cap in row.items():
            cap = 0
            if coupling_name in mat[master_name]:
                cap = mat[master_name][coupling_name]
            abserr += abs(cap - ref_cap)
            abssum += abs(ref_cap)
    if abssum == 0:
        return 0.0, selfcap_err
    return abserr / abssum, selfcap_err


def run_rwcap_analysis(n_runs, test_cases=['case1', 'case2'], n_cores=16):
    all_results = {}
    
    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"RUNNING ANALYSIS FOR {case.upper()} WITH {n_cores} CORES")

        ref_file = os.path.join(REFERENCE_FOLDER, f"{case}.dspf")
        if not os.path.exists(ref_file):
            ref_file = os.path.join(REFERENCE_FOLDER, f"{case}.spice")
        ref_capmat, _, _ = parse_result(ref_file)
        if ref_capmat is None:
            print(f"Warning: Reference file for {case} not found in {REFERENCE_FOLDER}")
        
        print(f"{'='*60}")

        accuracy_setting = TESTCASE_SETTING[case] if case in TESTCASE_SETTING else "-p 0.01"
        
        output_file = f"{case}.cap3d.out"
        log_file = f"{case}.cap3d.log"
        capacitances = [None] * n_runs
        execution_times = [np.nan] * n_runs
        cpu_times = [np.nan] * n_runs  # NEW: Add CPU time storage
        relative_errors = [None] * n_runs
        matrix_errors = [np.nan] * n_runs
        num_walks_list = [None] * n_runs
        gpu_stats_list = [None] * n_runs 
        hops_per_walk_list = [None] * n_runs
        total_tasks_list = [None] * n_runs
        
        print(f"Running rwcap command {n_runs} times for {case} (convergence criterion=\'{accuracy_setting}\', cores={n_cores})...")
        print("Extracting capacitance, timing, GPU stats, and computing relative error")

        print("-" * 60)
        
        for i in range(n_runs):
            print(f"Run {i+1}/{n_runs}:")
            sys.stdout.flush()
            
            try:
                command = [
                    "./bin/rwcap", "--walk-type", "SS", 
                    "-f", os.path.join(TESTCASE_FOLDER, f"{case}.cap3d"),
                    "-n", f"{n_cores}",
                ] + accuracy_setting.split(' ')

                subprocess.run(command, capture_output=False, text=True)
                
                if os.path.exists(output_file):
                    capmat, num_walks, hops_per_walk = parse_result(output_file)
                    num_walks_list[i] = num_walks
                    hops_per_walk_list[i] = hops_per_walk
                    
                    # Calculate total tasks
                    if num_walks and hops_per_walk:
                        total_tasks_list[i] = int(num_walks * hops_per_walk)
                    
                    rel_error, selfcap_err = compute_error(capmat, ref_capmat)
                    master_list = sorted(list(selfcap_err.keys()))
                    if master_list:
                        cap_str = '\t'.join([f'{capmat[x][x]:.6e}F' for x in master_list[:3]])
                        err_str = '\t'.join([f'{selfcap_err[x]*100:.2f}%' for x in master_list[:3]])
                        print(f"  Self-capacitance error: {err_str}")
                    print(f"  Matrix error: {rel_error*100:.2f}%")
                    if len(master_list):
                        capacitances[i] = {x: capmat[x][x] for x in master_list}
                        relative_errors[i] = selfcap_err
                        matrix_errors[i] = rel_error

                    with open(output_file, 'r') as f:
                        content = f.read()
                    
                    # MODIFIED: Updated regex to capture both elapsed and CPU time
                    time_match = re.search(r'Elapsed time:\s*([\d\.]+)sec,\s*CPU time:\s*([\d\.]+)sec', content)
                    if time_match:
                        elapsed_time = float(time_match.group(1))
                        cpu_time = float(time_match.group(2))
                        execution_times[i] = elapsed_time
                        cpu_times[i] = cpu_time  # NEW: Store CPU time
                        print(f"  Elapsed time: {elapsed_time:.3f}s, CPU time: {cpu_time:.3f}s")
                    else:
                        print(f"  Warning: Could not extract timing info from run {i+1}")
                    
                    # Parse GPU stats from log file
                    if os.path.exists(log_file):
                        gpu_stats = parse_gpu_stats_from_log(log_file)
                        gpu_stats_list[i] = gpu_stats
                    
                    sys.stdout.flush()

                else:
                    print(f"  Error: Output file {output_file} not found after run {i+1}")
            except subprocess.TimeoutExpired:
                print(f"  Error: Run {i+1} timed out")
            except Exception as e:
                print(f"  Error in run {i+1}: {e}")
            print()

        successful_gpu_stats = [s for s in gpu_stats_list if s is not None and s['green_total_tasks'] is not None]

        successful_times = [t for t in execution_times if not np.isnan(t)]
        successful_cpu_times = [t for t in cpu_times if not np.isnan(t)]  # NEW: Filter successful CPU times
        successful_matrix_errors = [e for e in matrix_errors if not np.isnan(e)]
        successful_num_walks = [w for w in num_walks_list if w is not None]
        
        successful_capacitances = []
        successful_selfcap_errors = []
        master_for_stats = None
        if any(c is not None for c in capacitances):
            first_valid_cap_dict = next(c for c in capacitances if c is not None)
            if first_valid_cap_dict:
                master_for_stats = sorted(list(first_valid_cap_dict.keys()))[0]
                for i, cap_dict in enumerate(capacitances):
                    if cap_dict and master_for_stats in cap_dict:
                        successful_capacitances.append(cap_dict[master_for_stats])
                        if relative_errors[i] and master_for_stats in relative_errors[i]:
                            successful_selfcap_errors.append(relative_errors[i][master_for_stats])

        successful_total_tasks = [t for t in total_tasks_list if t is not None]
        
        all_results[case] = {
            'capacitances': successful_capacitances,
            'execution_times': successful_times,
            'cpu_times': successful_cpu_times,  # NEW: Add CPU times to results
            'matrix_errors': successful_matrix_errors,
            'num_walks': successful_num_walks,
            'selfcap_errors': successful_selfcap_errors,
            'gpu_stats': successful_gpu_stats,
            'total_tasks': successful_total_tasks,
            'master_name': master_for_stats,
            'n_cores': n_cores
        }
        
        print("-" * 60)
        print(f"RESULTS ANALYSIS FOR {case.upper()} WITH {n_cores} CORES")
        print("-" * 60)
        
        print(f"\nIndividual results for {case} ({n_cores} cores):")
        # MODIFIED: Add CPU time column to individual results table
        print("Run | Capacitance      | Matrix Error | Elapsed Time | CPU Time     | Num Walks")
        print("-" * 80)
        
        for i in range(n_runs):
            run_num = i + 1
            cap_dict = capacitances[i]
            cap_val = cap_dict[master_for_stats] if cap_dict and master_for_stats in cap_dict else None
            cap_str = f"{cap_val:.6e}F" if cap_val is not None else "N/A"
            elapsed_time_val = execution_times[i]
            elapsed_time_str = f"{elapsed_time_val:.3f}s" if not np.isnan(elapsed_time_val) else "N/A"
            cpu_time_val = cpu_times[i]  # NEW: Get CPU time
            cpu_time_str = f"{cpu_time_val:.3f}s" if not np.isnan(cpu_time_val) else "N/A"  # NEW: Format CPU time
            error_val = matrix_errors[i]
            error_str = f"{error_val*100:.2f}%" if not np.isnan(error_val) else "N/A"
            walks_str = str(num_walks_list[i]) if num_walks_list[i] is not None else "N/A"
            print(f"{run_num:2d}  | {cap_str:>15} | {error_str:>12} | {elapsed_time_str:>8} | {cpu_time_str:>8} | {walks_str:>9}")
        
    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 rwcap_analysis.py <number_of_runs> [case1] [case2] ...")
        sys.exit(1)
    
    try:
        n_runs = int(sys.argv[1])
        if n_runs <= 0:
            raise ValueError("Number of runs must be positive")

        available_cases = [f.replace('.cap3d', '') for f in os.listdir(TESTCASE_FOLDER) if f.endswith('.cap3d')]

        if len(sys.argv) > 2:
            test_cases = []
            for case in sys.argv[2:]:
                if case == "all":
                    # Add all cases from case1 to case10
                    all_numbered_cases = [f"case{i}" for i in range(1, 11)]
                    for numbered_case in all_numbered_cases:
                        if numbered_case in available_cases and numbered_case not in test_cases:
                            test_cases.append(numbered_case)
                    print(f"Added all numbered cases (case1-case10): {[c for c in all_numbered_cases if c in available_cases]}")
                elif case in available_cases:
                    if case not in test_cases:  # Avoid duplicates
                        test_cases.append(case)
                else:
                    print(f"Warning: Unknown test case '{case}'. Available: {available_cases}")
            
            if not test_cases:
                print("No valid test cases specified. Exiting.")
                sys.exit(1)
        else:
            test_cases = available_cases
        
        print(f"Running analysis for: {', '.join(test_cases)}")
        
        # Define core counts to test
        core_counts = [16, 8, 4, 2, 1]
        all_results_by_cores = {}
        
        # Run analysis for each core count
        for n_cores in core_counts:
            print(f"\n{'='*80}")
            print(f"STARTING ANALYSIS WITH {n_cores} CORES")
            print(f"{'='*80}")
            
            results = run_rwcap_analysis(n_runs, test_cases, n_cores)
            all_results_by_cores[n_cores] = results

            print(f"\n{'='*60}")
            print("OVERALL COMPARISON")
            print(f"{'='*60}")
            
            print(f"\nSummary Table:")
            # MODIFIED: Add CPU time column to summary table
            header = "| Case  | Cap (μ±σ)            | Elapsed Time (μ±σ) | CPU Time (μ±σ)     | MatrixErr (μ±σ) | SelfCapErr (μ±σ)  | Walks (μ±σ)       | Total Tasks (μ±σ)   | GPU Tasks (Grad/Green)      | GPU Batch (Grad/Green)      |"
            separator = "|-------|----------------------|--------------------|--------------------|-----------------|--------------------|-------------------|---------------------|-----------------------------|-----------------------------|"
            print(header)
            print(separator)
            
            # NEW: Write summary table to file as well
            summary_filename = f"rwcap_summary_{n_cores}cores.txt"
            with open(summary_filename, "w") as sum_f:
                sum_f.write(f"Summary Table for {n_cores} cores\n")
                sum_f.write("=" * 60 + "\n")
                sum_f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                sum_f.write(f"Total runs per case: {n_runs}\n")
                sum_f.write(f"Test cases: {', '.join(test_cases)}\n\n")
                sum_f.write(header + "\n")
                sum_f.write(separator + "\n")
                
                # For each test case, print detailed comparison across all core counts
                for case in test_cases:
                    for n_cores_inner in core_counts:
                        if n_cores_inner in all_results_by_cores and case in all_results_by_cores[n_cores_inner]:
                            data = all_results_by_cores[n_cores_inner][case]
                            
                            case_label = f"{case}({n_cores_inner}c)"
                            
                            if data['capacitances']:
                                cap_mean = np.mean(data['capacitances'])
                                cap_std = np.std(data['capacitances'], ddof=1)
                                cap_str = f"{cap_mean:.3e}±{cap_std:.2e}"
                            else:
                                cap_str = "N/A"
                            
                            if data['execution_times']:
                                time_mean = np.mean(data['execution_times'])
                                time_std = np.std(data['execution_times'], ddof=1)
                                time_str = f"{time_mean:.1f}±{time_std:.1f}s"
                            else:
                                time_str = "N/A"
                            
                            # NEW: Add CPU time formatting
                            if data['cpu_times']:
                                cpu_time_mean = np.mean(data['cpu_times'])
                                cpu_time_std = np.std(data['cpu_times'], ddof=1)
                                cpu_time_str = f"{cpu_time_mean:.1f}±{cpu_time_std:.1f}s"
                            else:
                                cpu_time_str = "N/A"
                            
                            if data['matrix_errors']:
                                err_mean = np.mean(data['matrix_errors']) * 100
                                err_std = np.std(data['matrix_errors'], ddof=1) * 100
                                err_str = f"{err_mean:.1f}±{err_std:.1f}%"
                            else:
                                err_str = "N/A"
                            
                            if data['selfcap_errors']:
                                selfcap_mean = np.mean(data['selfcap_errors']) * 100
                                selfcap_std = np.std(data['selfcap_errors'], ddof=1) * 100
                                selfcap_str = f"{selfcap_mean:.1f}±{selfcap_std:.1f}%"
                            else:
                                selfcap_str = "N/A"

                            if data['total_tasks']:
                                tasks_mean = np.mean(data['total_tasks'])
                                if len(data['total_tasks']) > 1:
                                    tasks_std = np.std(data['total_tasks'], ddof=1)
                                    tasks_str = f"{tasks_mean:.0f}±{tasks_std:.0f}"
                                else:
                                    tasks_str = f"{tasks_mean:.0f}"
                            else:
                                tasks_str = "N/A"
                            
                            if data['num_walks']:
                                walks_mean = np.mean(data['num_walks'])
                                if len(data['num_walks']) > 1:
                                    walks_std = np.std(data['num_walks'], ddof=1)
                                    walks_str = f"{walks_mean:.0f}±{walks_std:.0f}"
                                else:
                                    walks_str = f"{walks_mean:.0f}"
                            else:
                                walks_str = "N/A"
                            
                            # Add GPU stats with gradient/green order and std dev
                            if data['gpu_stats'] and len(data['gpu_stats']) > 0:
                                green_tasks = [s['green_total_tasks'] for s in data['gpu_stats'] if s.get('green_total_tasks') is not None]
                                green_batch = [s['green_avg_batch_size'] for s in data['gpu_stats'] if s.get('green_avg_batch_size') is not None]
                                grad_tasks = [s['gradient_total_tasks'] for s in data['gpu_stats'] if s.get('gradient_total_tasks') is not None]
                                grad_batch = [s['gradient_avg_batch_size'] for s in data['gpu_stats'] if s.get('gradient_avg_batch_size') is not None]
                                
                                # Build GPU tasks string with std dev
                                if grad_tasks:
                                    grad_tasks_mean = np.mean(grad_tasks)
                                    if len(grad_tasks) > 1:
                                        grad_tasks_std = np.std(grad_tasks, ddof=1)
                                        grad_tasks_str = f"{grad_tasks_mean:.0f}±{grad_tasks_std:.0f}"
                                    else:
                                        grad_tasks_str = f"{grad_tasks_mean:.0f}"
                                else:
                                    grad_tasks_str = "0"
                                    
                                if green_tasks:
                                    green_tasks_mean = np.mean(green_tasks)
                                    if len(green_tasks) > 1:
                                        green_tasks_std = np.std(green_tasks, ddof=1)
                                        green_tasks_str = f"{green_tasks_mean:.0f}±{green_tasks_std:.0f}"
                                    else:
                                        green_tasks_str = f"{green_tasks_mean:.0f}"
                                else:
                                    green_tasks_str = "0"
                                    
                                gpu_tasks_str = f"{grad_tasks_str}/{green_tasks_str}"
                                
                                # Build GPU batch string with std dev
                                if grad_batch:
                                    grad_batch_mean = np.mean(grad_batch)
                                    if len(grad_batch) > 1:
                                        grad_batch_std = np.std(grad_batch, ddof=1)
                                        grad_batch_str = f"{grad_batch_mean:.0f}±{grad_batch_std:.0f}"
                                    else:
                                        grad_batch_str = f"{grad_batch_mean:.0f}"
                                else:
                                    grad_batch_str = "0"
                                    
                                if green_batch:
                                    green_batch_mean = np.mean(green_batch)
                                    if len(green_batch) > 1:
                                        green_batch_std = np.std(green_batch, ddof=1)
                                        green_batch_str = f"{green_batch_mean:.0f}±{green_batch_std:.0f}"
                                    else:
                                        green_batch_str = f"{green_batch_mean:.0f}"
                                else:
                                    green_batch_str = "0"
                                    
                                gpu_batch_str = f"{grad_batch_str}/{green_batch_str}"
                            else:
                                gpu_tasks_str = "N/A"
                                gpu_batch_str = "N/A"
                            
                            # MODIFIED: Add CPU time to the format string
                            line = f"| {case_label:<5} | {cap_str:<20} | {time_str:<18} | {cpu_time_str:<18} | {err_str:<15} | {selfcap_str:<18} | {walks_str:<17} | {tasks_str:<19} | {gpu_tasks_str:<27} | {gpu_batch_str:<27} |"
                            print(line)
                            sum_f.write(line + "\n")


        # Generate comprehensive summary
        print(f"\n{'='*80}")
        print("COMPREHENSIVE COMPARISON ACROSS ALL CORE COUNTS")
        print(f"{'='*80}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"rwcap_scaling_analysis_{timestamp}.txt"
        
        with open(results_filename, "w") as f:
            f.write("RWCap Core Scaling Analysis Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total runs per case/core combination: {n_runs}\n")
            f.write(f"Test cases: {', '.join(test_cases)}\n")
            f.write(f"Core counts tested: {', '.join(map(str, core_counts))}\n\n")
            
           
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Test case folder not found at {TESTCASE_FOLDER}")
        sys.exit(1)
import os
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

def find_notebook(filename, search_root):
    """
    Recursively search for the notebook file in the project structure.
    """
    for path in search_root.rglob(filename):
        return path
    return None

def execute_notebook(notebook_path, project_root):
    """
    Executes a notebook with the project root set as the working directory.
    """
    print(f"--- Processing: {notebook_path.name} ---")
    print(f"Location: {notebook_path}")
    start_time = time.time()

    try:
        # 1. Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # 2. Configure execution (timeout=-1 for RL training)
        executor = ExecutePreprocessor(timeout=-1, kernel_name='python3')

        # 3. Path Mounting: Set 'path' to project_root so notebooks can find 
        # local modules (like llava_client.py) and save to 'logs_and_results'.
        executor.preprocess(nb, {'metadata': {'path': str(project_root)}})

        duration = time.time() - start_time
        print(f"--- Finished {notebook_path.name} in {duration:.2f}s ---\n")
        return True

    except Exception as e:
        print(f"Execution Error in {notebook_path.name}: {e}")
        return False

def main():
    # Detect Project Root (where main.py is located)
    project_root = Path(__file__).parent.absolute()
    
    # Ensure the results directory exists for GitHub users
    log_dir = project_root / "logs_and_results"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
        print(f"Created directory: {log_dir}")

    # Define the required execution sequence
    pipeline_files = [
        "baseline_mountain_car.ipynb",
        "ours_vlm_ppo.ipynb",
        "final_report.ipynb"
    ]

    print("======================================================")
    print("RL-VLM PROJECT PIPELINE")
    print(f"Project Root: {project_root}")
    print("======================================================\n")

    for target_name in pipeline_files:
        # Dynamically find the file (fixes the 'algorithms' folder issue)
        nb_path = find_notebook(target_name, project_root)
        
        if nb_path:
            success = execute_notebook(nb_path, project_root)
            if not success:
                print(f"CRITICAL ERROR: Pipeline stopped at {target_name}")
                return
        else:
            print(f"FILE NOT FOUND: {target_name}")
            print("Please ensure the notebook exists in the project subfolders.")
            return

    print("======================================================")
    print("SUCCESS: Full workflow completed.")
    print(f"Outputs saved in: {log_dir}")
    print("======================================================")

if __name__ == "__main__":
    main()
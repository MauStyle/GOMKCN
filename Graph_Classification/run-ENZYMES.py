import subprocess
import itertools
from collections import OrderedDict
import multiprocessing

Parameter = OrderedDict([
    ('seed', [0]),
    ('epoch', [200]),
    ('lr', [0.01]),
    ('wd', [0]),
    ('batch', [128]),
    ('result', ['result']),
    ('root', ['dataset']),
    ('dataset', ['ENZYMES']),
    ('hop', [1]),
    ('size', [6]),
    ('dim', [(64,8)]),
    ('fold', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ('step', [2]),
    ('hidden', [16]),
    ('norm', [True]),
    ('actv', [True]),
    ('pool', ['mean']),
    ('device', ['cuda:0']),
])

def run_experiment(params):
    command, param_list = ['python', 'main.py'], []
    for key, value in params.items():
        if isinstance(value, bool):
            param_list.append(f"-{key}") if value else None
        else:
            param_list.extend([f"-{key}", str(value)])
    full_command = command + param_list

    print(f"Running: {' '.join(full_command)}")
    result = subprocess.run(full_command, capture_output=False, text=False)

    param_info = ", ".join(f"{k}={v}" for k, v in params.items())

    if result.stdout:
        print(f"Output for {param_info}:\n{result.stdout}")
    if result.stderr:
        print(f"Errors for {param_info}:\n{result.stderr}")

def run_experiment_wrapper(params):
    try:
        run_experiment(params)
    except Exception as e:
        print(f"Error running experiment with params {params}: {e}")

if __name__ == '__main__':
    keys = Parameter.keys()
    values = Parameter.values()

    all_combinations = list(itertools.product(*values))
    all_params = [dict(zip(keys, combination)) for combination in all_combinations]

    num_processes = 1

    print(f"Total experiments: {len(all_params)}")
    print(f"Using {num_processes} processes")

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_experiment_wrapper, all_params)

    print("\nAll runs completed.")
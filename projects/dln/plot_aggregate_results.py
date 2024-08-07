import os
import pandas as pd


from dln.__main__ import plot_results

RESULTS_DIR = "dln/results"

def get_subdirectories(root_dir):
    subdirectories = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            subdirectories.append(full_path)
    return subdirectories



if __name__ == "__main__":

    all_results = pd.concat((pd.read_csv(os.path.join(path, 'results.csv')) for path in get_subdirectories(RESULTS_DIR)))

    plot_results(all_results, log_plot=True)

    
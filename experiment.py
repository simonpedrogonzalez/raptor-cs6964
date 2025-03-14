from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH, RESULTS_PATH
import pandas as pd
import numpy as np
from dataclasses import dataclass
import gc
import tqdm
import psutil
from profile import profile
from reference import reference_method, compare_results
import datetime
from file_utils import describe_raster_file, describe_vector_file, validate_raster_vector_compatibility
import json
import seaborn as sns
import matplotlib.pyplot as plt

class Experiment:
    def __init__(self, raster_path, vector_path, func, reps=1, stats=['mean']):
        self.raster_path = raster_path
        self.vector_path = vector_path
        self.func = func
        self.reps = reps
        self.stats = stats

    def _reset(self):
        gc.collect()
        # Clears both page cache and inodes/dentries cache assuming linux os
        # os.system("sync; echo 3 > /proc/sys/vm/drop_caches")
        # I dont think random seeds needs to be setted since all algos are deterministic

    def _write_results(self, metric_list):

        print("Writing results...")

        now_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        exp_name = f"exp_{now_string}.json"
        metrics_df = pd.DataFrame(metric_list)
        metrics_summary = metrics_df.describe()

        # boxplot per metric in the same figure
        n_metrics = len(metrics_df.columns)
        fig, axs = plt.subplots(1, n_metrics, figsize=(15, 5))
        for i, col in enumerate(metrics_df.columns):
            sns.boxplot(data=metrics_df[col], ax=axs[i])
            axs[i].set_title(col)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_PATH}/boxplot_{now_string}.png")
        plt.clf()
        # lineplot per metric
        fig, axs = plt.subplots(n_metrics, 1, figsize=(5, 10))
        for i, col in enumerate(metrics_df.columns):
            sns.lineplot(data=metrics_df[col], ax=axs[i])
            # axs[i].set_title(col)
            # remove x label from all but the last plot
            if i < n_metrics - 1:
                axs[i].set_xlabel("")
        plt.tight_layout()
        # remove vertical space between plots
        plt.subplots_adjust(hspace=0)
        plt.savefig(f"{RESULTS_PATH}/lineplot_{now_string}.png")
        plt.clf()
        

        vector_summary = describe_vector_file(self.vector_path)
        raster_summary = describe_raster_file(self.raster_path)

        res = {
            "func": self.func.__name__,
            "reps": self.reps,
            "stats": self.stats,
            "raster": raster_summary,
            "vector": vector_summary,
            "metrics": metrics_summary.to_dict()
        }

        # write a json file with the results
        res_path = f"{RESULTS_PATH}/{exp_name}"
        with open(res_path, "w") as f:
            json.dump(res, f)
        print(f"Results written to {res_path}")

    def run(self):

        print(f"Starting experiment with {self.reps} repetitions")
        print(f"Function: {self.func.__name__}")
        print(f"Raster: {self.raster_path}")
        print(f"Vector: {self.vector_path}")
        print(f"Stats: {self.stats}")

        validate_raster_vector_compatibility(self.raster_path, self.vector_path)
        truth = reference_method(self.raster_path, self.vector_path, self.stats)

        metric_list = []
        
        for i in tqdm.tqdm(range(self.reps)):
            self._reset()
            
            # I'm wrapping the func call with the profiler
            result, metrics = profile(self.func)(self.raster_path, self.vector_path, self.stats)
            
            correct, errors = compare_results(truth, result)
            if not correct:
                raise ValueError(f"Error in result: {errors}")
            
            metric_list.append(metrics)

        self._write_results(metric_list)
        return

    

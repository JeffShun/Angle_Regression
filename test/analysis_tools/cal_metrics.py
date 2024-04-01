"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional
import os
import numpy as np
from runstats import Statistics
import pandas as pd

def abs_error(pred, label):
    return abs(pred - label)/np.pi*180

METRIC_FUNCS = dict(
    AbsError=abs_error,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics = {metric: Statistics() for metric in metric_funcs}
        self.metrics_data = {metric:[] for metric in metric_funcs}

    def push(self, pid, pred, label):
        for metric, func in METRIC_FUNCS.items():
            val = func(pred, label)
            self.metrics[metric].push(val)
            self.metrics_data[metric].append((pid, val))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}
    

    def save(self, save_dir):
        os.makedirs(save_dir,exist_ok=True)
        df = pd.DataFrame()
        # 遍历数据字典，将每种评价方式的数据添加为DataFrame的一列
        for method, values in self.metrics_data.items():
            labels, scores = zip(*values)
            df[method] = scores
        df['pid'] = labels
        df = df[['pid'] + [col for col in df.columns if col != 'pid']]
        csv_file_path = save_dir + "/metrics.csv"
        df.to_csv(csv_file_path, index=False)


    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
            for name in metric_names
        )

def evaluate(args):
    metrics = Metrics(METRIC_FUNCS)
    for sample in args.data_path.iterdir():
        pid = str(sample).split("\\")[-1].replace("npz","")
        data = np.load(sample, allow_pickle=True)
        pred = data['pred']
        label = data['label']
        metrics.push(pid, pred, label)

    return metrics


if __name__ == "__main__":
    
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data_path", type=pathlib.Path,
        default=pathlib.Path("./data/output/raw_result"),
    )
    args = parser.parse_args()
    metrics = evaluate(args)
    metrics.save("./data/output/metric_data")
    print(metrics)

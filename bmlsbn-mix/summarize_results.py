import torch

import math
import numpy as np
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--only-mean', action='store_true')
args = parser.parse_args()

total = []
for imb in ["0.005", "0.01", "0.02", "0.1"]:
    if args.verbose:
        print(f"imbalance ratio: {imb}")
    if args.method == 'dual':
        filepaths = sorted(Path(args.dirpath).glob(f'*dual*/*{imb}*/log.txt'))
    else:
        filepaths = sorted(Path(args.dirpath).glob(f'**/*None*{imb}*/log.txt'))

    ts_accs = []
    for filepath in filepaths:
        # load checkpoint
        with open(filepath, 'r') as f:
            lines = f.readlines()
            ts_acc = float(lines[-1].strip())
            if args.verbose:
                print(ts_acc)
            #if ts_acc < 75:
            #    continue
            ts_accs.append(ts_acc)

    if args.verbose:
        print("{:.2f} {:.2f}".format(np.mean(ts_accs), np.std(ts_accs)))
    total.append(np.mean(ts_accs))
    if not args.only_mean:
        total.append(np.std(ts_accs))

for t in total:
    print("{:.2f}".format(t), end=" ")
print()


from pandas_read_prof import read_torch_prof as rp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import pandas as pd
from argparse import ArgumentParser
import sys

argp = ArgumentParser()
argp.add_argument('--output_channels', type=int, default=None)
argp.add_argument('--kernel_size', type=int, default=None)
argp.add_argument('--input_channels', type=int, default=None)
argp.add_argument('--batch_size', type=int, default=None)
argp.add_argument('--image_size', type=int, default=None)
argp.add_argument('--epoch', type=int, default=None)
argp.add_argument('--xaxis', type=str, required=True, help='the name of the column graphed on the xaxis, if the variable is "values" then unique values for each column are printed')
argp.add_argument('--dir', type=str, required=True, help='the directory where the output files are')

def main():
    args = argp.parse_args(sys.argv[1:])
    df = read_data(args.dir)

    if args.xaxis == 'values':
        for c in df.columns:
            print(f'{c} : {df[c].unique()}')
        raise SystemExit

    mask = np.array([True] * len(df))
    for k, v in args._get_kwargs():
        if v == None or k == 'xaxis':
            continue

        if (df[k] == v).sum() > 0:
            mask &= (df[k] == v)
        else:
            raise Exception(f"value {v} doesn't appear in column {k}")

    df.loc[mask].plot(args.xaxis, 'time', kind='scatter')
    plt.show()



def read_data(dir_name):
    name_re = re.compile(".*_(\d+).txt")

    p = Path(dir_name)
    files = list(p.iterdir())

    profiles = []
    for f in files:
        t = rp(f)
        if len(t[0]) == 0:
            continue


        t[0]['epoch'] = int(name_re.match(f.name).group(1))
        profiles.append(t)

    

    agg_data = {k : [] for k in profiles[0][0].keys()}

    agg_data['time'] = []
    for prof in profiles:
        for k,v in prof[0].items():
            agg_data[k].append(v)
        agg_data['time'].append(prof[1]['total_time'].max())

    return pd.DataFrame(agg_data)


    


if __name__ == '__main__':
    main()

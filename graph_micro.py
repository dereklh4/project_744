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
#epochs 2 to 6 are averaged currently
#argp.add_argument('--epoch', type=int, default=None)
argp.add_argument('--xaxis', type=str, required=True, help='the name of the column graphed on the xaxis, if the variable is "values" then unique values for each column are printed')
argp.add_argument('--dir', type=str, required=True, help='the directory where the output files are')

def main():
    args = argp.parse_args(sys.argv[1:])
    df = None
    try:
        df = pd.read_pickle('df.pickle')
        print("sucessfully loaded pickled dataframe")
    except Exception as e:
        print("Unable to load pickled dataframe")
        print(e)
        df = read_data(args.dir)
        df.to_pickle('df.pickle')
        print('saving dataframe in df.pickle')



    if args.xaxis == 'values':
        for c in df.columns:
            print(f'{c} : {df[c].unique()}')
        raise SystemExit
    title = []
    mask = np.array([True] * len(df))
    for k, v in args._get_kwargs():
        if v == None or k == 'xaxis' or k == 'dir':
            continue
        title.append(f'{k} = {v}')
        if (df[k] == v).sum() > 0:
            mask &= (df[k] == v)
        else:
            raise Exception(f"value {v} doesn't appear in column {k}")
    # plot forward
    line_df = df.loc[mask]
    fdf = line_df[line_df.forward]
    bdf = line_df[~line_df.forward]

    fig, ax = plt.subplots(figsize=(8,8))
    #ax.plot(fdf[args.xaxis], fdf['time'], color='r', label='forward')
    ax.plot(fdf[args.xaxis], fdf['time'] / fdf['batch_size'], 'r.-', label='per example forward')
    #ax.plot(bdf[args.xaxis], bdf['time'], color='b', label='backward')
    ax.plot(bdf[args.xaxis], bdf['time'] / bdf['batch_size'], 'b.-', label='per example backward')
    ax.set_xticks(bdf[args.xaxis])

    ax.set_xlabel(args.xaxis)
    ax.set_ylabel('time (ms)')
    ax.set_title(" ".join(title))
    ax.legend()
    fig.tight_layout()
    tlt = "".join(title).replace(' = ', '')
    #fig.savefig('./graphs/' + tlt + "_graph.pdf")   
    plt.show()



def read_data(dir_name):
    name_re = re.compile(".*_(\d+).txt")

    p = Path(dir_name)
    files = list(p.iterdir())

    profiles = []
    for f in files:
        epoch = int(name_re.match(f.name).group(1))
        if epoch < 4:
            continue
        t = rp(f)
        if len(t[0]) == 0:
            continue
        
        #ignore the first two runs
        
        
        profiles.append(t)

    
    # meta data keys
    agg_data = {k : [] for k in profiles[0][0].keys()}
    print(agg_data)

    agg_data['time'] = []
    agg_data['forward'] = []
    for prof in profiles:
        for k,v in prof[0].items():
            agg_data[k].append(v)
        agg_data['forward'].append(prof[1].at[0, 'forward'])
        agg_data['time'].append(prof[1]['total_time'].max())
    
    df = pd.DataFrame(agg_data).groupby( ['output_channels', 'kernel_size', 'input_channels', 'batch_size', 'image_size', 'forward'], as_index=False).mean()

    df['per_example'] = df.time / df.batch_size
    return df
    




    


if __name__ == '__main__':
    main()

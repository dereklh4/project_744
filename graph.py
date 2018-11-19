from pandas_read_prof import read_prof as rp
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Useage: python3 graph.py <file name> <column> <y label> <top k> <output file>')
        raise SystemExit

    f = sys.argv[1]
    col = sys.argv[2]
    ylab = sys.argv[3]
    topk = int(sys.argv[4])
    output_file = sys.argv[5]
    df, meta = rp(f, 'KB', 'ms', 'm')
    srted = df.sort_values(col)[-topk:]
    subplt = srted.plot('node name', col, kind='bar', figsize=(8, 8))
    subplt.axes.set_ylabel(ylab)
    subplt.axes.get_legend().remove()
    subplt.figure.tight_layout()
    subplt.figure.savefig(output_file)

    plt.show()






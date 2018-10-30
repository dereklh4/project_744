import pandas as pd
import re
import sys

#################################################################
# units that the output will be converted into
MEM_UNIT = 'MB'
T_UNIT = 'ms'
#################################################################


mem_units  ={ 
    'GB': 2**30,
    'MB': 2**20,
    'KB': 2**10,
    'B': 1
}

time_units = {
    'hr' : 3600,
    'min': 60,
    's': 1,
    'sec': 1,
    'ms': 1/(10**3),
    'us': 1/(10**6)
}

line_re = re.compile('(\S+)\s+(\d*\.?\d+)(\S+)\s+\(.*\),\s+(\d*\.?\d+)(\S+)\s+\(.*\),\s+(\d*\.?\d+)(\S+)\s+\(.*\),\s+(\d*\.?\d+)(\S+)\s+\(.*\).*$')




def convert_mem(val, unit):
    return float(val) * (mem_units[unit.upper()]) / mem_units[MEM_UNIT.upper()]

def convert_time(val, unit):
    return float(val) * (time_units[unit.lower()]) / time_units[T_UNIT.lower()]
    
def parse_line(line):
    m =line_re.match(line)
    if m == None:
        print('error parsing line %s' % line)
        return None
    else:
        res = [m.group(1)]
        res.append(convert_mem(m.group(2), m.group(3)))
        for i in range(4, 9, 2):
            res.append(convert_time(m.group(i), m.group(i+1)))

        return tuple(res)


    
def main():
    if len(sys.argv) < 2:
        print('Usage : python pandas_read_prof.py <file name>')
        exit()
    fname = sys.argv[1]
    lines_gen = (l for l in open(fname).readlines())
    while 'Profile:' not in next(lines_gen):
        pass

    columns = [s.strip() for s in next(lines_gen).split('|')]
    data_dict = {c : [] for c in columns}
    for line in lines_gen:
        res = parse_line(line)
        if res != None:
            for idx in range(len(res)):
                data_dict[columns[idx]].append(res[idx])

    for k in list(data_dict.keys()):
        if len(data_dict[k]) == 0:
            del data_dict[k]

    df = pd.DataFrame(data_dict)

    print(df)


if __name__ == '__main__':
    main()

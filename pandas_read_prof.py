from __future__ import division
import pandas as pd
import numpy as np
import re

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

flops_units = {
    'b' : 1000,
    'm' : 1,
    'k': 10**(-3),
    'empty': 10**(-6)
}


line_sub = re.compile('\(\d*\.?\d+%, \d*\.?\d+%\)')
cell_split = re.compile('')

val_unit_re = re.compile('(\d*\.?\d+)([a-zA-Z]+)')

def val_unit(cell):
    #print(cell)
    cell  = cell.strip()
    m = val_unit_re.search(cell)
    if m != None:
        return (m.group(1), m.group(2))
    else:
        return (cell, 'empty')
    
def convert_mem(val, unit, desired_unit):
    return float(val) * (mem_units[unit.upper()]) / mem_units[desired_unit.upper()]

def convert_time(val, unit, desired_unit):
    return float(val) * (time_units[unit.lower()]) / time_units[desired_unit.lower()]
    
def convert_flops(val, unit, desired_unit):
    return float(val) * (flops_units[unit.lower()]) / flops_units[desired_unit.lower()]

def parse_line(line, mem_unit, t_unit, flops_unit):
    line = line_sub.sub('', line)
    cells = line.split(',')
    first = cells.pop(0)
    first, second = first.split()
    cells.insert(0,second)
    cells.insert(0,first)
    print(cells)

    mem_cells = [1,2,3]
    t_cells = [4,5,6]
    for idx in mem_cells:
        val, unit = val_unit(cells[idx])
        cells[idx]  = convert_mem(val, unit, mem_unit)

    for idx in t_cells:
        val, unit = val_unit(cells[idx])
        cells[idx]  = convert_time(val, unit, t_unit)

    #params
    cells[7] = cells[7].split()[0].strip()
    #ops 
    cell = cells[8].split()[0].strip()
    print(cell)
    if cell != '0':
        val, unit = val_unit(cell)
        cells[8] = convert_flops(val, unit, flops_unit)
    else:
        cells[8] = 0.0

    return tuple(cells)

    


    
def read_prof(fname, mem_unit, t_unit, flops_unit):
    meta = {
	'mem_unit' : mem_unit,
	'time_unit': t_unit,
	'flops_unit': flops_unit,
	'time' : fname[:-4],
	'filename' : fname
    }

    lines_gen = (l for l in open(fname).readlines())
    line = next(lines_gen)
    while len(line) > 1:
        k, v = eval(line)
        meta[k] = v
        line = next(lines_gen)

 
    while 'Profile:' not in next(lines_gen):
        pass

    columns = [s.strip() for s in next(lines_gen).split('|')]
    data_dict = {c : [] for c in columns}
    for line in lines_gen:
        res = parse_line(line, mem_unit, t_unit, flops_unit)
        if res != None:
            for idx in range(len(res)):
                data_dict[columns[idx]].append(res[idx])

    for k in list(data_dict.keys()):
        if len(data_dict[k]) == 0:
            del data_dict[k]

    df = pd.DataFrame(data_dict)
    return (df, meta)

def read_torch_prof(fname, raw=False):
    SCALE = 1000
    lines = iter(open(fname).readlines())
    l = next(lines)
    meta = {}
    while '------' not in l:
        k, v = eval(l)
        meta[k] = v
        l = next(lines)
    next(lines)
    next(lines)
    #get the first 3 items in each row
    sre = re.compile('\\s\\s+')
    tuples = [tuple([s.strip() for s in sre.split(l)[:3]]) for l in lines]

    #return a raw dataframe
    if raw:
        data = {
            'op_name' : [t[0] for t in tuples],
            'cpu_time' : [float(t[1][:-2]) / SCALE for t in tuples],
            'gpu_time' : [float(t[2][:-2]) / SCALE for t in tuples]
        }
        return pd.DataFrame(data)
        

    raw_data = {}
    op_names = {t[0] for t in tuples}
    for op in op_names:
        raw_data[op] = {
            'cpu_time' : [],
            'gpu_time' :[],
        }

    for t in tuples:
        #convert to miliseconds
        raw_data[t[0]]['cpu_time'].append(float( t[1][:-2] ) / SCALE)
        raw_data[t[0]]['gpu_time'].append(float( t[2][:-2] ) / SCALE)
    

    data = {
        'op_name' : [],
        'cpu_std_dev' : [],
        'cpu_avg' : [],
        'cpu_total' : [],
        'gpu_std_dev' : [],
        'gpu_avg' : [],
        'gpu_total' : [],
        'ncalls' : [],
        'total_time' : []
    }

    for op in op_names:
        row = raw_data[op]
        row['cpu_time'] = np.array(row['cpu_time'])
        row['gpu_time'] = np.array(row['gpu_time'])

        data['op_name'].append(op)

        data['cpu_std_dev'].append(row['cpu_time'].std())
        data['cpu_avg'].append(row['cpu_time'].mean())
        data['cpu_total'].append(row['cpu_time'].sum())

        data['gpu_std_dev'].append(row['gpu_time'].std())
        data['gpu_avg'].append(row['gpu_time'].mean())
        data['gpu_total'].append(row['gpu_time'].sum())

        data['ncalls'].append(len(row['gpu_time']))
        data['total_time'].append(row['gpu_time'].sum() + row['cpu_time'].sum() )


    return pd.DataFrame(data)

        



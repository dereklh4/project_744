import pandas as pd
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
    'k': 10**(-3)
}


line_sub = re.compile('\(\d*\.?\d+%, \d*\.?\d+%\)')
cell_split = re.compile('')

val_unit_re = re.compile('(\d*\.?\d+)(\S+)')

def val_unit(cell):
    print(cell)
    cell  = cell.strip()
    m = val_unit_re.search(cell)
    return (m.group(1), m.group(2))
    
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
    if cell != '0':
        val, unit = val_unit(cell)
        cells[8] = convert_flops(val, unit, flops_unit)
    else:
        cells[8] = 0.0

    return tuple(cells)

    


    
def read_prof(fname, mem_unit, t_unit, flops_unit):
    lines_gen = (l for l in open(fname).readlines())
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
    return df



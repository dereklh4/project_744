import subprocess
import pprint

sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

out_str = sp.communicate()
out_list = out_str[0].split('\n')

out_dict = {}

for item in out_list:
    try:
        key, val = item.split(':')
        key, val = key.strip(), val.strip()
        out_dict[key] = val
    except:
        pass
aa = out_dict["Used GPU Memory"]
print (aa.split()[0])
print (out_dict["Used GPU Memory"])

import json
import pandas as pd
import ipdb

f = open("results_microbenchmarks.txt")
for line in f:
	#ipdb.set_trace()
	data_dict = json.loads(line)
	df = pd.DataFrame.from_dict(data_dict,orient="columns")
	print(df)


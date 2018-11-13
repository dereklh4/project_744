

To download data:
	$ python download_and_convert_data.py \
	    --dataset_name=cifar10 \
	    --dataset_dir=data/cifar10

	$ python download_and_convert_data.py \
	    --dataset_name=flowers \
	    --dataset_dir=data/flowers
	# note that the scripts expect datasets to be in ./data/<dataset_name>

to remove the logs (needed between runs):
	# deletes /tmp/train_logs/*
	$ bash rm_logs.sh 


To Run:
	#this only needs to be done once, all scripts expect this environment var to be set
	$ export BATCH_SIZE=<desired batch size>
	$ ./<model_name>/{cifar, flowers}.sh <NUMBER OF STEPS>

	results are written to 
	./results/<model_name>/<time scripts was run>.txt
	

Reading back the results into pandas data frame:

	>>> from pandas_read_prof import read_prof
	>>> df, meta = read_prof(<FILENAME>, <MEM_UNIT>, <TIME_UNIT>, <FLOPS_UNIT>)
	# meta is a dictionary of the all the flags set during that run
	# as well as the units in the dataframe (i.e. memory_unit, time_unit, float_op unit)
	# df will be the actual dataframe, indexed by op_name


	valid units are:
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
	

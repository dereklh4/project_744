import tensorflow as tf
import sys
from tensorflow.python.platform import gfile
import numpy as np
from tensorflow.python.util import compat
from datetime import datetime
import os

try:
	os.mkdir("./profiler")
except:
	pass

try:
	os.mkdir("./timelines")
except:
	pass

model_filename =sys.argv[1]
with gfile.FastGFile(model_filename, 'rb') as f:

	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')

current_time = datetime.now().strftime("%Y%m%d-%H%M")

builder = tf.profiler.ProfileOptionBuilder
opts = (builder(builder.time_and_memory()).
				with_step(-1). # with -1, should compute the average of all registered steps.
				with_file_output("profiler/test-" + current_time + ".txt").
				select(["peak_bytes","micros"]).order_by("micros").
				build())

with tf.contrib.tfprof.ProfileContext('profiler',trace_steps=[],dump_steps=[]) as pctx:
	with tf.Session() as sess:
		flat_tensor = sess.graph.get_tensor_by_name('pool_3:0')

		for i in range(6):
			# Enable tracing for next session.run.
			pctx.trace_next_step()
			pctx.dump_next_step()
			
			out_val = sess.run(flat_tensor, {'DecodeJpeg:0': np.random.ranf((244,244,3))})
			
			# Dump the profile
			pctx.profiler.profile_operations(options=opts)



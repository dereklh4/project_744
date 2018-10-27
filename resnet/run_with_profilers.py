import tensorflow as tf
import sys
from tensorflow.python.platform import gfile
import numpy as np
# from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from datetime import datetime
import os

#option 1 - use timeline
from tensorflow.python.client import timeline
#https://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow/37774470#37774470

#option 2 - put in tensorboard runtime statistics - includes memory and compute time

#option 3 - profiler
#https://gist.github.com/notoraptor/4cfeaaf2ab24ebce59ac727f389096fa

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

with tf.Session() as sess:
	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	flat_tensor = sess.graph.get_tensor_by_name('pool_3:0')

	# Option 2: Put in tensorboard and can see memory, compute time on individual nodes
	summary_writer = tf.summary.FileWriter('log/' + current_time, graph=tf.get_default_graph())

	# Option 3: Profiler
	profiler = tf.profiler.Profiler(sess.graph)

	for i in range(5):
		out_val = sess.run(flat_tensor, {'DecodeJpeg:0': np.random.ranf((244,244,3))},options=run_options,run_metadata=run_metadata)

		#option 2
		summary_writer.add_run_metadata(run_metadata, 'step%d' % i)

		#option 3
		profiler.add_step(i, run_metadata)

	# Option 1: Create the Timeline object, and write it to a json. View at chrome://tracing
	tl = timeline.Timeline(run_metadata.step_stats)
	ctf = tl.generate_chrome_trace_format()
	with open('timelines/timeline-' + current_time + '.json', 'w') as f: #this will just be the last run. Need to be appending if want all runs
		f.write(ctf)

	#option 3
	option_builder = tf.profiler.ProfileOptionBuilder
	
	#opts = (tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
	opts = (option_builder(option_builder.time_and_memory()).
			with_step(-1). # with -1, should compute the average of all registered steps.
			with_file_output("profiler/test-" + current_time + ".txt").
			select(['micros','bytes','occurrence']).order_by('micros').
			build())
	profiler.profile_operations(options=opts)

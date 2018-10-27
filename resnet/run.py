import tensorflow as tf
import sys
from tensorflow.python.platform import gfile
import numpy as np
# from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

model_filename =sys.argv[1]
with gfile.FastGFile(model_filename, 'rb') as f:

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
                    
    
with tf.Session() as sess:
    flat_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    for i in range(10):
        out_val = sess.run(flat_tensor, {'DecodeJpeg:0': np.random.ranf((244,244,3))})
    tf.profiler.profile(sess.graph)

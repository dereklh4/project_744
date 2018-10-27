
TENSORFLOW=//home/derek/.conda/pkgs/tensorflow-base-1.11.0-mkl_py36h3c3e929_0/lib/python3.6/site-packages/tensorflow

#bazel build -c opt $TENSORFLOW/tools/benchmark
#&& 
bazel-bin/tensorflow/tools/benchmark/benchmark_model \
--graph=/tmp/tensorflow_inception_graph.pb \
--input_layer="Mul" --input_layer_shape="1,299,299,3" \
--input_layer_type="float" --output_layer="softmax:0" \
--show_run_order=false --show_time=false \
--show_memory=false --show_summary=true \
--show_flops=true --logtostderr

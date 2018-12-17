output_channels="32" #  8 12 16 20 24 32"
kernel_size="1" # 3 5 7"
input_channels="16" #"4 8 12 16 32" 
batch_size="4 8 16 32 48 64 96"
for o in $output_channels
do 
	for i in $input_channels
	do
		for b in $batch_size
		do
			python3 graph_micro.py --xaxis  input_channels --kernel_size 5 --output_channels $o --batch_size $b  --dir ./jetson_results/micro_gpu
		done
	done
done

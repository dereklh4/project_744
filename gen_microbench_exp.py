
batch_range = [32,64,128]
input_channels_range = [3,32,64,128,256,512]
image_size_range = [256]
output_channels_range = [32,64,128,256,512]
kernel_size_range = (1,8,2) #start, stop, step


output_file = open("experiments.txt","w+")

c = 0
for v1 in batch_range:
	for v2 in input_channels_range:
		for v3 in image_size_range:
			for v4 in output_channels_range:
				for v5 in range(kernel_size_range[0],kernel_size_range[1],kernel_size_range[2]):
					output_file.write(",".join(map(str,[v1,v2,v3,v4,v5])) + "\n")
					c += 1
output_file.close()
print("Generated " + str(c) + " experiments")


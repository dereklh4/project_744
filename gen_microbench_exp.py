
#start,stop,step
batch_range = (1,256,16)
input_channels_range = (1,512,16)
image_size_range = (28,256,16)
output_channels_range = (1,512,16)
kernel_size_range = (1,7,2)


output_file = open("experiments.txt","w+")

c = 0
for v1 in range(batch_range[0],batch_range[1],batch_range[2]):
	for v2 in range(input_channels_range[0],input_channels_range[1],input_channels_range[2]):
		for v3 in range(image_size_range[0],image_size_range[1],image_size_range[2]):
			for v4 in range(output_channels_range[0],output_channels_range[1],output_channels_range[2]):
				for v5 in range(kernel_size_range[0],kernel_size_range[1],kernel_size_range[2]):
					output_file.write(",".join(map(str,[v1,v2,v3,v4,v5])) + "\n")
					c += 1
output_file.close()
print("Generated " + str(c) + " experiments")


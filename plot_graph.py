import json
import bokeh
from bokeh.layouts import column
from bokeh.plotting import figure, output_file, show

file_list = ['20pc_two_parts_stats.json', '40pc_two_parts_stats.json',
             '60pc_two_parts_stats.json',
             '100pc_two_parts_stats.json', '20pc_all_parts_stats.json',
             '40pc_all_parts_stats.json',
             '60pc_all_parts_stats.json', '100pc_all_parts_stats.json']
color_list = ["red", "blue", "green", "purple", "black", "orange", "pink",
              "olive","yellow"]
output_file("mnist_random_training.html")
loss_plot = figure(width=1500, height=500, title="Loss vs steps",
                   x_axis_label="Step Number", y_axis_label="loss value")
accuracy_plot = figure(width=1500, height=500, title="Accuracy vs epochs",
                       x_axis_label = "Epoch Number", y_axis_label= "Accuracy")
compression_ratio = figure(width=500, height=500, 
                           title="Time per epoch vs Training Setting",
                           x_axis_label= "Digits after decimal",
                           y_axis_label="Compression Ratio")

compress_val_list = list()
for idx, f in enumerate(file_list):
    print (f)
    with open(f, 'r') as in_file:
        file_data = json.load(in_file)
        loss_vals = file_data.get('loss_value')
        accuracy_vals = file_data.get('accuracy')
        compression_vals = file_data.get('Time per epoch')
        # import ipdb; ipdb.set_trace()
        x_axis = range(len(loss_vals))
        loss_plot.circle(x_axis, loss_vals, color=color_list[idx],
                         legend=f.rsplit('_', 1)[0])
        loss_plot.line(x_axis, loss_vals, line_color=color_list[idx],
                       legend=f.rsplit('_',1)[0])
        
        x_axis = range(len(accuracy_vals))
        accuracy_plot.circle(x_axis, accuracy_vals, color=color_list[idx],
                             legend=f.rsplit('_',1)[0])
        accuracy_plot.line(x_axis, accuracy_vals, line_color=color_list[idx],
                           legend=f.rsplit('_',1)[0])
        try:
            compress_avg = sum(compression_vals)/len(compression_vals)
        except:
            break
        compress_val_list.append(compress_avg)
x_axis_vals = [cc.rsplit('_',1)[0] for cc in file_list]
# import ipdb; ipdb.set_trace()

compression_ratio = figure(width=1500, height=500, 
                           title="Time per epoch vs Training Setting",
                           x_axis_label= "Digits after decimal",
                           y_axis_label="Compression Ratio",
                           x_range=x_axis_vals)
compression_ratio.circle(x_axis_vals, compress_val_list, size=9)

p = column(loss_plot, accuracy_plot, compression_ratio)
show(p)



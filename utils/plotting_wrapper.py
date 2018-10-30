import io
import imageio
import numpy as np
import seaborn as sns
from PIL import Image
from IPython import display
import matplotlib.pyplot as plt


def show_file_image(filename):

    plt.figure(figsize=(12, 12))
    plt.imshow(plt.imread(filename))
    plt.axis('off')


def read_pil_image_from_plt(plt):

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)


def create_gif(img_generator, cmap=plt.cm.viridis, gif_name="./.__gif_sample.gif", fps=10,
               figsize=(4, 4), title=None):

    fig = plt.figure(figsize=figsize)
    with imageio.get_writer(gif_name, mode='I', fps=fps) as writer:

        for img in img_generator():
            # Clear canvas
            plt.gca().cla()
            ax = plt.gca()
            # Draw
            plt.imshow(img, cmap=cmap)
            plt.xticks([])
            plt.yticks([])
            plt.title(title)
            # Wait to draw - only for online visualization
            display.clear_output(wait=True)
            display.display(plt.gcf())
            # Append to GIF
            img = read_pil_image_from_plt(plt)
            writer.append_data(np.array(img))
        plt.clf()
    return

# Adapted from: https://gist.github.com/extrospective/0f4fe69304184d813f982035d9684452
# Actually it's the same function with a minor modification in how to select palatte colors


def stacked_bar_chart(pivoted_df, stack_vals, level_values_field, chart_title, x_label, y_label, palette="pastel"):
    #
    # stacked_bar_chart: draws and saves a barchart figure to filename
    #
    # pivoted_df: dataframe which has been pivoted so columns correspond to the values to be plotted
    # stack_vals: the column names in pivoted_df to plot
    # level_values_field: column in the dataframe which has the values to be plotted along the x axis (typically time dimension)
    # chart_title: how to title chart
    # x_label: label for x axis
    # y_label: label for y axis
    # filename: full path filename to save file
    # color1: first color in spectrum for stacked bars
    # color2: last color in spectrum for stacked bars; routine will select colors from color1 to color2 evenly spaced
    #
    # Implementation: based on (http://randyzwitch.com/creating-stacked-bar-chart-seaborn/; https://gist.github.com/randyzwitch/b71d47e0d380a1a6bef9)
    # this routine draws overlapping rectangles, starting with a full bar reaching the highest point (sum of all values), and then the next shorter bar
    # and so on until the last bar is drawn.  These are drawn largest to smallest with overlap so the visual effect is that the last drawn bar is the
    # bottom of the stack and in effect the smallest rectangle drawn.
    #
    # Here "largest" and "smallest" refer to relationship to foreground, with largest in the back (and tallest) and smallest in front (and shortest).
    # This says nothing about which part of the bar appear large or small after overlap.
    #
    color_spectrum = sns.color_palette(palette, len(stack_vals))
    plt.clf()
    #
    stack_total_column = 'Stack_subtotal_xyz'  # placeholder name which should not exist in pivoted_df
    bar_num = 0
    legend_rectangles = []
    legend_names = []
    for bar_part in stack_vals:    # for every item in the stack we need to compute a rectangle
        stack_color = color_spectrum[bar_num]  # get_hex_l ensures full hex code of color
        sub_count = 0
        pivoted_df[stack_total_column] = 0
        stack_value = ""
        # for every item in the stack we create a new subset [stack_total_column] of 1 to N of the sub values
        for stack_value in stack_vals:
            pivoted_df[stack_total_column] += pivoted_df[stack_value]  # sum up total
            sub_count += 1
            # we skip out after a certain number of stack values
            if sub_count >= len(stack_vals) - bar_num:
                break
        # now we have set the subtotal and can plot the bar.  reminder: each bar is overalpped by smaller subsequent bars starting from y=0 axis
        bar_plot = sns.barplot(data=pivoted_df, x=pivoted_df.index.get_level_values(level_values_field),
                               y=stack_total_column, color=stack_color)
        legend_rectangles.append(plt.Rectangle((0, 0), 1, 1, fc=stack_color, edgecolor='none'))
        # the "last" stack_value is the name of that part of the stack
        legend_names.append(stack_value)
        bar_num += 1
    l = plt.legend(legend_rectangles, legend_names, loc=2, ncol=1, prop={'size': 12})
    l.draw_frame(False)
    bar_plot.set(xlabel=x_label, ylabel=y_label)
    plt.tight_layout()
    plt.title(chart_title)
    sns.despine(left=True)
    # plt.savefig(filename)

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

    
# Bokeh Google Maps
from bokeh.io import output_file, show, output_notebook, save, export_png
from bokeh.models import ColumnDataSource, GMapOptions
from bokeh.plotting import gmap
output_notebook()

def show_img(filename):
    plt.figure(figsize=(12,12))
    plt.imshow(plt.imread(filename))
    plt.axis('off')

def show_p(p):
    show(p)
    
def visualize_trajectory(lats, lngs, actions=None,
                         file=None, render=False, p=None,
                         fill_col=None, draw_col="black", fill_alpha=1., title="",
                         font_size="6pt", dot_size=5, line_col=None, line_width=1.,
                         plot_endpt=False, endpt_beta=3, endpt_line_width=1., 
                         endpt_alpha=0.8, endpt_plot_fn=None, endpt_line_col=None,
                         show_grid_fn=None,
                         key=None, zoom=14, W=800, H=800):
    
    lats = np.asarray(lats)
    lngs = np.asarray(lngs)
    
    if p is None:
        map_options = GMapOptions(lat=np.median(lats), lng=np.median(lngs), map_type="satellite", zoom=zoom)
        p = gmap(key, map_options, title=title, plot_width=W, plot_height=H)
    
    if show_grid_fn is not None:
        show_grid_fn(p, np.vstack((lats, lngs)).transpose())
        
    if plot_endpt:
        
        endpt_source = ColumnDataSource(dict(x=[lngs[0], lngs[-1]], 
                                        y=[lats[0], lats[-1]], 
                                        colors=["green", "red"]))
        
        endpt_line_col = fill_col if endpt_line_col is None else endpt_line_col
        if endpt_plot_fn is not None:
            endpt_plot_fn("x", "y", size=endpt_beta*dot_size, line_color=endpt_line_col, 
                          line_width=endpt_line_width*endpt_beta, fill_color="colors", 
                          fill_alpha=int(endpt_alpha*endpt_beta), source=endpt_source)
        else:
            p.circle("x", "y", size=endpt_beta*dot_size, line_color=endpt_line_col, 
                     line_width=int(endpt_alpha*endpt_beta), fill_color="colors", 
                     fill_alpha=endpt_alpha, source=endpt_source)
        
    if actions is not None:
        source = ColumnDataSource(
            data=dict(lat=lats,
                      lon=lngs,
                      action=actions)
            )
        p.text(x="lon", y="lat", text="action", text_font_size=font_size, 
                       text_color=draw_col, source=source)
    else:
        source = ColumnDataSource(
            data=dict(lat=lats,
                      lon=lngs)
            )
        p.circle(x="lon", y="lat", size=dot_size, fill_color=fill_col, line_color=draw_col, 
                 fill_alpha=fill_alpha, source=source)
        
        if line_col:
            p.line(x="lon", y="lat", line_color=line_col, line_width=line_width, source=source)
    
    if file is not None:
        plt.close('all')
        export_png(p, filename=file)
        if render: show_img(file)
    else:    
        if render: show(p)
    return p
    
def show_grid(p, lat_lvls, lng_lvls, lat_min, lat_max, lng_min, lng_max, 
              vlines=True, hlines=True, line_color="white", alpha=0.7):
    
    if vlines:
        for lat in lat_lvls:
            if lat_min <= lat <= lat_max:
                p.line( (lng_min, lng_max),  (lat, lat), line_color=line_color, line_alpha=alpha)
    
    if hlines:
        for lng in lng_lvls:
            if lng_min <= lng <= lng_max:
                p.line( (lng, lng),  (lat_min, lat_max), line_color=line_color, line_alpha=alpha)
                
                
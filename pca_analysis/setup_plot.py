import matplotlib.pyplot as plt 
from matplotlib import font_manager as fm 
from matplotlib.pyplot import gca
import matplotlib as mpl
from cycler import cycler
import math
import numpy as np

linestyle_tuple = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot'),
     
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
     
     ]

linestyle_dict = {k : v for k,v in linestyle_tuple}

def setup_global():
    font_entry = fm.FontEntry(
        fname = './gillsans.ttf',
        name='gill-sans')

    # set font
    fm.fontManager.ttflist.insert(0, font_entry) 
    mpl.rcParams['font.family'] = font_entry.name 

    mpl.use('Agg')
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 4
    mpl.rcParams['hatch.linewidth'] = 0.1

def setup_local(axis=None):
    if axis is None:
        plt.clf()
        axis = gca()
    
    axis.yaxis.grid(linestyle='dotted', which="both")
    axis.spines['left'].set_color('#606060')
    axis.spines['bottom'].set_color('#606060')
    global_cycler = cycler(color=get_colors()[:len(get_linestyles())]) + cycler(linestyle=get_linestyles()) 
    axis.set_prop_cycle(global_cycler)

def set_aspect_ratio(ratio=3/5, logx=None, logy=None, axis=None):
    if axis is None:
        axis = gca() 
    xleft, xright = axis.get_xlim()
    if logx is not None:
        xleft = math.log(xleft, logx)
        xright = math.log(xright, logx)
    ybottom, ytop = axis.get_ylim()
    if logy is not None:
        ytop = math.log(ytop, logy)
        print(ytop, ybottom)
        ybottom = math.log(ybottom, logy)
    axis.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

def get_colors():
    return ['#D55E00', 
            '#009E73',
            '#0072B2', 
            '#CC79A7', 
            '#000000', 
            '#E03A3D',
            '#F0E442',]

def get_hatches():
    return ['xxx', 'xxxxxx', '\\\\\\\\', '||||||','///////', '+']

def get_linestyles():
    return [
            linestyle_dict['solid'], 
            linestyle_dict['dotted'], 
            linestyle_dict['dashed'],
            linestyle_dict['dashdotted'],
            linestyle_dict['dashdotdotted'],
            linestyle_dict['densely dashdotted'],
            
    ]

def get_markers():
    return ['^', 's', 'o', 'd', 'x']

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    #ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    #ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
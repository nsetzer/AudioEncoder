
import os
import sys
import matplotlib
import matplotlib.gridspec as gridspec
import argparse
import platform
import subprocess
import multiprocessing
import json

from collections import namedtuple, defaultdict

def read_json(path):
    """
    read dictionaries out of a json file

    the json file can contain a single dictionary, in which
    case a sinle plot will be generated

    Alternatively a list of dictionaries can be read in, which
    will cause multiple plots to be rendered
    """

    with open(path, "r") as rf:
        data= json.load(rf)

        if not isinstance(data,list):
            data = [data,]

        return data

def parse_and_set_ylim(plt_, csv):
    """ set the y-axis limits

    'ymin,ymax' => set both upper and lower limit
    ',ymax' => set only the upper limit
    'ymin' => set only the lower limit
    """
    if csv:
        if "," in csv:
            _ymin, _ymax = csv.split(",")
        else:
            _ymin = csv
            _ymax = ""
        if _ymin:
            plt_.ylim(ymin=float(_ymin))
        if _ymin:
            plt_.ylim(ymax=float(_ymax))

def plot_systeminfo(plt_, sysinfo):
    info0 = "%s\nMemory: %s" % (sysinfo.system, sysinfo.mem)
    info1 = ("Cpu Info:\nProcessor: %s\n" +
            "Physical: %d\nLogical: %d\nFrequency: %s") % sysinfo.cpu
    info2 = "Cache Info:\nL1i: %s\nL1d: %s\nL2: %s\nL3: %s" % sysinfo.cache
    plt_.text(.00, .5, info0, va="top")
    plt_.text(.33, .5, info1, va="top")
    plt_.text(.66, .5, info2, va="top")
    plt_.axis('off')

def plot_singleline(plt_, data, secondary_view, ylim):
    """plot a single line from the given data

    data is a dictionary containing the metadata describing a plot
    """

    name = data.get('graph', 'unknown')

    handles = []
    graph_data = data['data2'] if secondary_view else data['data']
    x,y = zip(*graph_data)
    plt_.plot(x, y, marker='o')

    plt_.legend(bbox_to_anchor=(1.0, 1.0), framealpha=0.25)

    xlabel = data.get('xlabel', "")
    ylabel = data.get('ylabel2' if secondary_view else 'ylabel', "")

    yunits = data.get('yunits2' if secondary_view else 'yunits', "")
    if yunits:
        yunits = " (%s)" % yunits

    xunits = data.get('xunits', "")
    if xunits:
        xunits = " (%s)" % xunits

    title = data.get('title', '%s vs %s' % (ylabel, xlabel))

    if 'ymin' in data:
        plt_.ylim(ymin=float(data['ymin']))

    if 'ymax' in data:
        plt_.ylim(ymax=float(data['ymax']))

    parse_and_set_ylim(plt_, ylim)

    plt_.title(title)
    plt_.xlabel(xlabel + xunits)
    plt_.ylabel(ylabel + yunits)

    if name != "unknown":
        plt_.text(0.025, 0.95, name,
             horizontalalignment='left',
             verticalalignment='top',
             fontsize=14,
             transform=plt_.gca().transAxes)

def plot_multiline(plt_, data, secondary_view, ylim):
    """plot multiple labeled lines from the given data

    data is a dictionary containing the metadata describing a plot
        the key is a string, the label for the line
        the value is a list of 2-tuples. the x and y coords of the plot
    """

    graph_data = data['data2'] if secondary_view else data['data']
    handles = []
    for label, points in graph_data.items():
        x,y = zip(*points)
        handle, = plt_.plot(x, y, label=label, marker='o')
        handles.append(handle)

    plt_.legend(bbox_to_anchor=(1.0, 1.0), framealpha=0.25)

    xlabel = data.get('xlabel', "")
    ylabel = data.get('ylabel2' if secondary_view else 'ylabel', "")

    yunits = data.get('yunits2' if secondary_view else 'yunits', "")
    if yunits:
        yunits = " (%s)" % yunits

    xunits = data.get('xunits', "")
    if xunits:
        xunits = " (%s)" % xunits

    title = data.get('title', '%s vs %s' % (ylabel, xlabel))

    parse_and_set_ylim(plt_, ylim)

    plt_.title(title)
    plt_.xlabel(xlabel + xunits)
    plt_.ylabel(ylabel + yunits)

def plot_boxplot(plt_, data, secondary_view, ylim):
    """plot a series of boxplots from the given data

    data is a dictionary containing the metadata describing a plot
    """

    name = data.get('graph', 'unknown')
    results = data['data']

    # create a box plot with dummy data, fill in the correct values below
    box = plt_.boxplot([[0.0, ] for i in range(len(results))])

    if len(results) > 15:
        labels = list(range(1, len(results) + 1))
        ticks = list(range(min(labels) - 1, max(labels) + 1, 5))
        plt_.xticks(ticks, ticks)

    CL = []
    for i, item in enumerate(results):
        j = i * 2
        k = j + 1
        IQR = item['q75'] - item['q25']
        CL0 = item['mean'] - (1.5 * IQR)
        CL1 = item['mean'] + (1.5 * IQR)
        CL0 = max(item['min'], CL0)
        CL1 = min(item['max'], CL1)
        CL.append(CL0)
        CL.append(CL1)
        box['medians'][i].set_ydata([item['mean'], ])
        box['boxes'][i]._xy[[0, 1, 4], 1] = item['q25']
        box['boxes'][i]._xy[[2, 3], 1] = item['q75']
        box['whiskers'][j].set_ydata([item['q25'], CL0])
        box['whiskers'][k].set_ydata([item['q75'], CL1])
        box['caps'][j].set_ydata([CL0, CL0])
        box['caps'][k].set_ydata([CL1, CL1])

    xlabel = data.get('xlabel', "")
    ylabel = data.get('ylabel', "")

    yunits = data.get('yunits', "")
    if yunits:
        yunits = " (%s)" % yunits

    xunits = data.get('xunits', "")
    if xunits:
        xunits = " (%s)" % xunits

    title = data.get('title', '%s vs %s' % (ylabel, xlabel))

    ymin = min(CL)
    ymax = max(CL)
    ptp = ymax - ymin
    plt_.ylim((ymin - 0.1 * ptp, ymax + 0.1 * ptp))
    parse_and_set_ylim(plt_, ylim)

    # plt_.grid("lightgray", linestyle="--", axis="y")

    plt_.title(title)
    plt_.xlabel(xlabel + xunits)
    plt_.ylabel(ylabel + yunits)

    plt_.text(0.025, 0.95, name,
         horizontalalignment='left',
         verticalalignment='top',
         fontsize=14,
         transform=plt_.gca().transAxes)

def plot_main(plt_, data, secondary_view, ylim=""):
    """
    generates a plt given a dictionary of parameters

    plt_ is passed in as an argument so that this function can be imported
    into other modules.
    """
    mode = data['mode']
    if mode == "singleline":
        plot_singleline(plt_, data, secondary_view, ylim)
    elif mode == "multiline":
        plot_multiline(plt_, data, secondary_view, ylim)
    elif mode == "boxplot":
        plot_boxplot(plt_, data, secondary_view, ylim)
    else:
        raise Exception("unknown mode %s" % mode)

    if 'comment' in data:
        sys.stdout.write("\n\n%s\n" % data['comment'])

def main():

    parser = argparse.ArgumentParser(description="""
        Plot the performance results contained in a json file.
    """)

    parser.add_argument("--rtf", '--secondary', dest="secondary_view",
        action="store_true",
        help=""" display a secondary view for the plot.
         For batch performance results, this plots real time factor.
         A value of 1 is real time, a value of 2 is twice real time.
         Higher is better,
        """)

    parser.add_argument("--width", type=int, default=8, dest="width",
        help=""" width of plot in inches (8)""")

    parser.add_argument("--height", type=int, default=6, dest="height",
        help=""" height of plot in inches (6)""")

    parser.add_argument("--label", type=str, default="",
        help=""" provide a label to help identify where the data
        came from
        """)

    parser.add_argument("--ylim", type=str, default="",
        help="""set y axis limits. csv: min,max
        """)

    parser.add_argument("--sys-info", dest="sys_info",
        type=str, default=None,
        help="""filepath to read system information from
        """)

    parser.add_argument("data_path",
        help="path to a json results file")

    parser.add_argument("out_png", nargs='?', default=None,
        help=""" Optional. Save plot to a file.
        When not specified open a window to display the plot
        """)

    args = parser.parse_args()

    # ------------------------------------------------------------------------

    # enable headless use of matplotlib when saving to a file
    if args.out_png:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------------

    plt.figure(figsize=(args.width, args.height))

    # read the json file and return a list of dictionaries containing
    # the metadata used for generating a plot
    data = read_json(args.data_path)

    # determine the number of rows and columns of plots to display,
    # based on how many elements are in the list
    nrows = 1
    ncols = 2 if len(data) > 1 else 1
    nrows += max(0, (len(data)-2) // 2)

    height_ratios = [2,] * nrows

    gs = gridspec.GridSpec(nrows, ncols, height_ratios=height_ratios)

    for i in range(len(data)):
        if gridspec is not None:
            plt.subplot(gs[i])

        plot_main(plt, data[i], args.secondary_view, args.ylim)

    if args.out_png:
        plt.savefig(args.out_png)
    else:
        plt.show()

if __name__ == '__main__':
    main()

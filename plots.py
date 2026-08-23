"""
Basic plot settings for consistent styling across notebooks and papers.
"""

import matplotlib.pyplot as plt

LINEWIDTH = 1
MARKERSIZE = 3

# Single- and double-column widths for LaTeX
PT = 1./72.27
FIG_SIZES = {
    'onecol' : 246 * PT, 
    'twocol' : 510 * PT
    }

# Font sizes for labels, ticks, and legends
FONT_SIZES = {
    'labels' : 8,
    'ticks' : 8,
    'legend' : 7,
    'annotation' : 7
}

# Update fontsizes in matplotlib.
plt.rcParams.update({
    'axes.titlesize': FONT_SIZES['labels'],
    'axes.labelsize': FONT_SIZES['labels'],
    'xtick.labelsize': FONT_SIZES['ticks'],
    'ytick.labelsize': FONT_SIZES['ticks'],
    'legend.fontsize': FONT_SIZES['legend'],
    'font.size': FONT_SIZES['annotation'],
    'lines.linewidth': LINEWIDTH,
    'lines.markersize': MARKERSIZE,
    'grid.linewidth': LINEWIDTH/2,
})

"""Plotting functions for CNR results."""

import seaborn as sns
import matplotlib.pyplot as plt


def heatmap_cnr(d, cell_lines=None, figwidth=15):
    """Draw heatmap of d."""
    # Set up parameters
    if not cell_lines:
        cell_lines = list(d.keys())

    ncl = len(cell_lines)
    nrow, ncol = d[cell_lines[0]].shape
    fsize = (figwidth,  figwidth * nrow / (ncol * ncl + 3))

    fig, axn = plt.subplots(1, ncl, sharex=True, sharey=True, figsize=fsize)
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])
    cbar_ax = fig.add_axes([.91, .2, .02, .6])

    for i, ax in enumerate(axn.flat):
        cl = cell_lines[i]
        sns.heatmap(d[cl],
                    ax=ax,
                    cbar=i == 0,
                    linewidths=.3,
                    cbar_ax=None if i else cbar_ax)
        ax.set_title(cl, fontdict={'fontsize': 1.2 *
                                   figwidth, 'fontweight': "bold"})
        ax.tick_params(labelsize=figwidth * .8)

    # plt.tight_layout(rect=[0, 0, .9, 1])

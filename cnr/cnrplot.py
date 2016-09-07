"""Plotting functions for CNR results."""

import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


def heatmap_cnr(d, cell_lines=None, figwidth=15, annot=False):
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
                    annot=annot,
                    cbar_ax=None if i else cbar_ax)
        ax.set_title(cl, fontdict={'fontsize': 1.2 *
                                   figwidth, 'fontweight': "bold"})
        ax.tick_params(labelsize=figwidth * .8)


def graph_from_sol(sol, cl, widthfactor=10):
    """Generate graph from complete solution.

    This graph includes pertrubations as nodes, and edges from
    perturbations to the affected nodes.

    input:
    * sol: A CnrResult object.
    * cl: name of the cell line to use.

    output: An networkx DiGraph object
    """
    baseRp = 'rp_' + cl
    baseR = 'r_' + cl
    g = nx.DiGraph()
    # Go over all variables from the solution.
    for var, val in sol.vardict.items():
        # Treat perturbations and interactions seperately.
        if var.startswith(baseRp):
            assert len(var.split('_')) == 4
            pert = var.split('_')[2]
            node = var.split('_')[3]
            ind_name = '_'.join(['IrpDev', pert, node])
            # Color according to sign of perturbation.
            if val > 0:
                col = 'green'
                sign = 'positive'
            else:
                col = 'red'
                sign = 'negative'
            g.add_edge(pert, node, weight=val, color=col,
                       edgetype='perturbation',
                       penwidth=abs(val) * widthfactor,
                       deviation=sol.allowed_deviations[ind_name],
                       sign=sign)
        elif var.startswith(baseR):
            assert len(var.split('_')) == 4
            from_n = var.split('_')[3]
            to_n = var.split('_')[2]
            ind_name = '_'.join(['IDev', to_n, from_n])
            # Only add edge if in network
            if sol.vardict['_'.join(['I', to_n, from_n])]:
                # Color according to sign of perturbation.
                if val > 0:
                    col = 'green'
                    sign = 'positive'
                else:
                    col = 'red'
                    sign = 'negative'
                g.add_edge(from_n, to_n, weight=val, color=col,
                           edgetype='local_response',
                           penwidth=abs(val) * widthfactor,
                           deviation=sol.allowed_deviations[ind_name],
                           sign=sign)

    # Add node types for visualization purposes.
    nodes = sol.nodes
    node_types = dict()
    for n in g.nodes():
        if n in nodes:
            node_types[n] = 'protein'
        else:
            node_types[n] = 'perturbation'
    nx.set_node_attributes(g, 'type', node_types)

    return g

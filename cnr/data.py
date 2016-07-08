"""Classes to hold input data for Comparative Network Reconstruction."""

import numpy as np

# class PerturbationsAnnotation():
#     """Class to store information about perturbations. This information is
#     stored as a list of dictionaries, where each
#
#     Which nodes are affected? What is the mode of action? ect.
#
#     Parameters
#     ----------
#
#
#     Attributes
#     ----------
#
#
#     """
#
#     def __init__():


class PerturbationPanel:
    """" Class to contain data from a perturbation experiment.

    Parameters
    ----------
    nodes : tuple of strings
        Names of the measured nodes.

    rglob : dict of pandas.DataFrame
        Dataframe containg the results of a perturbation experiment. Rows
        correspond to nodes. Columns correspond to perturbation. Entries
        are the log-fold change in nodes to perturbation.

    perts_annot: dict
        Store information of the type of perturbations. Keys are perturbation
        names. Values are the corresponding PerturbationsAnnotation objects.

    Attributes
    ----------
    nodes : tuple of strings
        Names of the measured epitopes

    perts: list of lists
        Each element of the outer list represents a perturbation of which the
        response in measured. Each element of the inner list the combination of
        applied perturbations for that measurement.
        Example; [[EGFR, PLX], [AKTi]].

    rglob : dict of pandas.DataFrames
        Measured perturbation responses

    pert_annot
    """

    def __init__(self, nodes, perts, pert_annot=dict(), rglob=dict()):
        """Initialization."""
        # TO DO: generate mapping from rglob columns to perturbations.
        self._nodes = tuple(nodes)
        self._perts = perts
        self._pert_annot = pert_annot
        self._rglob = rglob.copy()
        self._cell_lines = list(rglob.keys())

    @property
    def nodes(self):
        """Get nodes, list of str indicating the measured epitopes."""
        return self._nodes

    @property
    def perts(self):
        """Get perts.

        A list of lists that annotates which perturbations were
        applied.
        """
        return self._perts

    @property
    def rglob(self):
        """Get dict of global response matrices (i.e. measurements)."""
        return self._rglob

    @property
    def cell_lines(self):
        """Get names of cell lines in the panel."""
        return self._cell_lines

    @property
    def pert_annot(self):
        """Get pert_annot.

        dict that maps applied perturbation to targeted nodes.
        """
        return self._pert_annot

    def add_cell_line(self, name, rglob):
        """Add response data from cell lines to PerturbationPanel.

        Parameters
        ----------
        name : str
            name of cell lines

        rglob : pandas.DataFrame
            Measured perturbation response
        """
        assert isinstance(name, str)
        self._check_consistency(rglob)
        self._cell_lines.append(name)
        self._rglob[name] = rglob.copy()

    def _check_consistency(self, rglob):
        assert tuple(rglob.index) == self._nodes, "Index don't match nodes"

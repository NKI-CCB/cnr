"""Classes to hold input data for Comparative Network Reconstruction."""

# import numpy as np
# import cnr.cplexutils

# class PerturbationAnnotation:
#     """Class to store information about perturbations. This information is
#     stored as a list of dictionaries, where each

#     Which nodes are affected? What is the mode of action? ect.

#     Parameters/Attributes
#     ----------
#     direct_targets: list of nodes

#     with_downstream_effect: list of nodes. Subset of direct targets

#     Attributes
#     ----------
#     direct_targets: list of nodes
#         Direct targets of the perturbation. If these are not measured, use
#         the first downstream nodes that are measured.
#     with_downsteam_effect: list of nodes
#         Nodes downstream of these targets may also be affected by this
#         perturbation. Used for e.g. inhibitors that affect the kinetic activity
#         of their target, causing a "perturbation" of the downstream nodes.
#     """

#     def __init__(self, direct_targets, with_downstream_effects=None):
#         assert isinstance(direct_targets, list)
#         if with_downstream_effects:
#             assert isinstance(with_downstream_effects, list)
#             assert set(with_downstream_effects).issubset(set(direct_targets))
#         self._direct_targets = direct_targets
#         self._with_downstream_effects = with_downstream_effects

#     @property
#     def direct_targets(self):
#         """Direct targets of perturbation."""
#         return self._direct_targets

#     @property
#     def with_downstream_effects(self):
#         """which perturbations have downstream effects."""
#         return self._with_downstream_effects


class PerturbationPanel:
    """" Class to contain data from a perturbation experiment.

    Parameters
    ----------
    nodes : tuple of strings
        Names of the measured nodes.

    perts: list of lists
        Each element of the outer list represents a perturbation of which the
        response in measured. Each element of the inner list the combination of
        applied perturbations for that measurement.
        Example; [[EGFR, PLX], [AKTi]].

    rglob : dict of pandas.DataFrame
        Dataframe containg the results of a perturbation experiment. Rows
        correspond to nodes. Columns correspond to perturbation. Entries
        are the log-fold change in nodes to perturbation.

    perts_annot: dict
        Store information of the type of perturbations. Keys are perturbation
        names, values are the corresponding targets. Each perturbation can only
        have one target; If it has multiple, use a different name for each
        perturbation/target name. E.g. suppose hgf affects both MEK and PI3K
        (though unobserved nodes, so directly), use hgfmek and hgfpi3k as
        different perturbations.

    ds_acting_perts: list
        Nodes downstream of these targets may also be affected by this
        perturbation. Used for e.g. inhibitors that affect the kinetic activity
        of their target, causing a "perturbation" of the downstream nodes.
    """

    def __init__(self, nodes, perts, pert_annot, ds_acting_perts,
                 rglob=None):
        """Initialization."""
        # TO DO: generate mapping from rglob columns to perturbations.
        self._nodes = tuple(nodes)
        self._perts = perts
        self._pert_annot = pert_annot.copy()
        self._ds_acting_perts = ds_acting_perts.copy()
        # if pert_annot:
        #     self._pert_annot = pert_annot.copy()
        # else:
        #     self._pert_annot = dict()
        if rglob:
            self._rglob = rglob.copy()
            self._cell_lines = list(rglob.keys())
        else:
            self._rglob = dict()
            self._cell_lines = []
        self._check_consistency()

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

    @cell_lines.setter
    def cell_lines(self, cell_lines):
        """Reorder cell line names."""
        assert set(cell_lines) == set(self.rglob.keys())
        self._cell_lines = cell_lines

    @property
    def pert_annot(self):
        """Get pert_annot.

        dict that maps applied perturbation to targeted nodes.
        """
        return self._pert_annot

    @property
    def ds_acting_perts(self): return self._ds_acting_perts

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
        self._cell_lines.append(name)
        self._rglob[name] = rglob.copy()
        self._check_consistency()

    def _check_consistency(self):
        assert isinstance(self._ds_acting_perts, list)
        assert set(self._ds_acting_perts).issubset(self._pert_annot.keys())
        assert set(self._pert_annot.values()).issubset(self._nodes), 'Not '+\
        ' all perturbation targets are nodes.'
        for pert in self._perts:
            assert isinstance(pert, list), 'Each element of perts should be list'
            assert set(pert).issubset(self._pert_annot.keys()), "Some \
                perturbations do not have direct target annotation:" +\
                str(set(pert) - set(self._pert_annot.keys()))

        for pert_name, pert_target in self.pert_annot.items():
            assert pert_target in self.nodes, "Direct target " +\
            str(pert_target) + " of " + str(pert_name) + " doesn't match nodes."

        for cell_line, rglob in self._rglob.items():
            assert tuple(rglob.index) == self._nodes, "Index of " + \
            str(cell_line) + " don't match nodes"
        
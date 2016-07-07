"""Classes and functions to store and analyze CNR optimization results."""

from cnr.data import PerturbationPanel
import numpy as np
import re

##############################################################################
#
# Helper functions


def get_rloc_from_cpx(cpx, nodes, prefix=None, solidx=0):
    """Returns: local response matrix, square np.array containing floats.

    Arguments:
    cpx:    Cplex object with feasible solution
    nodes:  Tuple of nodes in the reconstructed network
    Optional:
    prefix: str. Used if rloc is part of cell line panel
    solidx:  int. Used to select solution from solution pool
    """
    BASE = "r"
    if prefix:
        assert isinstance(prefix, str)
        BASE = "_".join([BASE, prefix])
    assert cpx.solution.is_primal_feasible()
    assert solidx < cpx.solution.pool.get_num()

    allvars = cpx.variables.get_names()
    vars_lst = [var for var in allvars if var.startswith(BASE + '_')]
    vars_vals = cpx.solution.pool.get_values(solidx, vars_lst)
    vars_dict = dict(zip(vars_lst, vars_vals))

    if 'r_i_i' in vars_dict.keys():
        del vars_dict['r_i_i']

    mat = -1. * np.identity(len(nodes))

    reg_sp = re.compile('(' + '|'.join(nodes) + ')_(' + '|'.join(nodes) + ')$')
    for key, value in vars_dict.items():
        sps = re.findall(reg_sp, key)
        assert len(sps) == 1, key + " has unexpected form"
        sps = sps[0]
        assert len(sps) == 2, key + " has unexpected form"
        i = nodes.index(sps[0])
        j = nodes.index(sps[1])
        mat[i][j] = value

    return mat


def get_deviations_from_cpx(cpx, nodes, prefix=None, solidx=0):
    """Returns: deviations matrix, square np.array containing floats.

    Deviation matrix only makes sense for cell line panel. Deviations are
    differences in local response coefficients or perturbation strengths
    from the cell line panel average.

    Arguments:
    cpx:    Cplex object with feasible solution
    nodes:  Tuple of nodes in the reconstructed network
    prefix: str. cell line name
    Optional:
    solidx:  int. Used to select solution from solution pool

    """
    base = "dev_r"
    if prefix:
        assert isinstance(prefix, str)
        base = "_".join([base, prefix])
    assert cpx.solution.is_primal_feasible()
    assert solidx < cpx.solution.pool.get_num()

    allvars = cpx.variables.get_names()
    vars_lst = [var for var in allvars if var.startswith(base + '_')]
    vars_vals = cpx.solution.pool.get_values(solidx, vars_lst)
    vars_dict = dict(zip(vars_lst, vars_vals))

    nn = len(nodes)
    mat = np.array([0.] * nn * nn).reshape(nn, nn)

    reg_sp = re.compile('(' + '|'.join(nodes) + ')_(' + '|'.join(nodes) + ')$')
    for key, value in vars_dict.items():
        sps = re.findall(reg_sp, key)
        assert len(sps) == 1, key + " has unexpected form"
        sps = sps[0]
        assert len(sps) == 2, key + " has unexpected form"
        i = nodes.index(sps[0])
        j = nodes.index(sps[1])
        mat[i][j] = value

    return mat


def get_rpert_from_cpx(cpx, rp_sym, solidx=0):
    """Construct rpert matrix from cplex solution.

    Input:
    - cpx
    - rp_sym:   matrix containing (names of) the perturbations form the
                perturbations experiment

    Output: matrix with the reconstructed strengths of the perturbations
    """
    assert cpx.solution.is_primal_feasible()

    allvars = cpx.variables.get_names()
    vars_lst = [var for var in allvars if var.startswith('rp_')]
    vars_vals = cpx.solution.pool.get_values(solidx, vars_lst)
    vars_dict = dict(zip(vars_lst, vars_vals))

    # print(rp_sym.all)

    nn = np.shape(rp_sym.all)[0]
    npert = np.shape(rp_sym.all)[1]

    mat = np.array([0.] * nn * npert).reshape(nn, npert)

    # for indices in zip(*np.nonzero(rp_sym)):
    #     ix = indices[0]
    #     jx = indices[1]
    #     mat[ix][jx] += vars_dict[str(rp_sym[ix][jx])]

    for indices in zip(*np.nonzero(rp_sym.inhib)):

        ix = indices[0]
        jx = indices[1]
        mat[ix][jx] += vars_dict[str(rp_sym.inhib[ix][jx])]

    for indices in zip(*np.nonzero(rp_sym.stim)):
        # print(indices)
        ix = indices[0]
        jx = indices[1]
        mat[ix][jx] += vars_dict[str(rp_sym.stim[ix][jx])]
    return mat

class CnrResult(PerturbationPanel):
    """Hold information of network reconstructions of a cell line panel.

    Parameters
    ----------
    p : CnrProblem instance

    """

    def __init__(self, p, solidx=0):
        """Initialilize object."""
        PerturbationPanel.__init__(
            self, nodes=p.nodes, perts=p.perts, pert_annot=p.pert_annot,
            rglob=p.rglob)

        # self.vardict = self._gen_vardict(p.cpx, solidx)
        # self.objectiveValue = p.cpx.solution.pool.get_objective_value(solidx)
        # self.imap = get_imap(p.cpx, self._nodes, solidx)
        self.rloc = self._gen_rloc_dict(p.cpx, solidx)
        self.rpert = self._gen_rpert_dict(p.cpx, p.rpert_dict, solidx=solidx)
        # self.bounds = self._get_bounds_from_cpx(p.cpx)
        # self.residuals = self._gen_residuals_dict(p.cpx, solidx)
        # self.mssr = self._gen_mssr_dict()
        # self.deviations = self._gen_deviations_dict(p.cpx, solidx)
        # self.allowed_deviations = get_allowed_deviations_from_cpx(p.cpx,
        #                                                           solidx)
        # self._rglobMeasured = p.get_rglob()
        # self.metaInfo = self._extract_metaInfo(p)
        # self.rpertSym = p.get_rpert_symbols()
        # self.rpert = self._gen_rpert_dict(p.cpx, self.rpertSym, solidx=solidx)
        # self.rglobPredicted = predict_response(self.rloc, self.rpert)
        # self.mean_sum_of_squares_deviations = \
        #     self._gen_sum_of_deviations_dict()

    def _gen_rloc_dict(self, cpx, solidx):
        rloc_dict = {}
        for cl in self.cell_lines:
            rloc_dict[cl] = get_rloc_from_cpx(cpx, self.nodes,
                                              prefix=cl, solidx=solidx)
        return rloc_dict

    def _gen_rpert_dict(self, cpx, rpSymDict, solidx):
        rp_dict = {}
        assert set(rpSymDict.keys()) == set(self.cell_lines)
        for key, value in rpSymDict.items():
            rp_dict[key] = get_rpert_from_cpx(cpx, value,
                                              solidx=solidx)


    # def _gen_deviations_dict(self, cpx, solidx):
    #     dev_dict = {}
    #     for cl in self._cell_lines:
    #         dev_dict[cl] = get_deviations_from_cpx(cpx, self._nodes,
    #                                                prefix=cl,
    #                                                solidx=solidx)
    #     return dev_dict

"""Classes and functions to store and analyze CNR optimization results."""

import numpy as np
import pandas as pd
import re
import cnr.cplexutils
from cnr.data import PerturbationPanel

########################################################################
#
# Public functions


def predict_response(rloc, pert):
    """Predict response of a perturbation based on local response matrix.

    Parameter:
    rloc : pandas.DataFrame or dict of pandas.DataFrames

    pert : pandas.DataFrame or dict of pandas.DataFrames
    """
    # Check if dimensions are OK
    if isinstance(rloc, dict):
        prediction = dict()
        assert isinstance(pert, dict)
        assert rloc.keys() == pert.keys()
        for key in rloc.keys():
            assert (np.shape(rloc[key])[0] ==
                    np.shape(rloc[key])[1] == len(pert[key]))
            prediction[key] = - np.dot(np.linalg.inv(rloc[key]), pert[key])

    else:
        assert np.shape(rloc)[0] == np.shape(rloc)[1] == len(pert)
        prediction = - np.dot(np.linalg.inv(rloc), pert)
    return prediction


########################################################################
#
# Classes

class CnrResult(PerturbationPanel):
    """Hold information of network reconstructions of a cell line panel.

    Parameters
    ----------
    p : CnrProblem instance

    solidx : int
        If p has solution pool, use to select solution.

    rloc : dict of pandas.DataFrame
        Reconstructed local response matrices


    Attributes
    ----------
    vardict : dict

    objective_value : float

    rloc : dict of pandas.DataFrames

    rpert :  dict of pandas.DataFrames



    """

    def __init__(self, p, solidx=0):
        """Initialilize object."""
        PerturbationPanel.__init__(
            self, nodes=p.nodes, perts=p.perts, pert_annot=p.pert_annot,
            rglob=p.rglob)

        self.vardict = self._gen_vardict(p.cpx, solidx)
        self.objective_value = p.cpx.solution.pool.get_objective_value(solidx)
        # self.imap = get_imap(p.cpx, self._nodes, solidx)
        self.rloc = self._gen_rloc_dict(p.cpx, solidx)
        self.rpert = self._gen_rpert_dict(p.cpx, p.rpert_dict, solidx=solidx)
        # self.rglob_predicted = predict_response(self.rloc, self.rpert)
        # self.bounds = self._get_bounds_from_cpx(p.cpx)
        # self.residuals = self._gen_residuals_dict(p.cpx, solidx)
        # self.mssr = self._gen_mssr_dict()
        # self.deviations = self._gen_deviations_dict(p.cpx, solidx)
        # self.allowed_deviations = get_allowed_deviations_from_cpx(p.cpx,
        #                                                           solidx)
        # self._rglobMeasured = p.get_rglob()
        # self.metaInfo = self._extract_metaInfo(p)
        # self.rpertSym = p.get_rpert_symbols()

        # self.mean_sum_of_squares_deviations = \
        #     self._gen_sum_of_deviations_dict()

    def _gen_vardict(self, cpx, solidx):
        """Retun dict with var: value as entries, for selected solution."""
        var_names = cpx.variables.get_names()
        var_vals = cpx.solution.pool.get_values(solidx, var_names)
        return dict(zip(var_names, var_vals))

    def _gen_rloc_dict(self, cpx, solidx):
        rloc_dict = {}
        for cl in self.cell_lines:
            rloc_dict[cl] = _get_rloc_from_cpx(cpx, self.nodes,
                                               prefix=cl, solidx=solidx)
        return rloc_dict

    def _gen_rpert_dict(self, cpx, rp_sym_dict, solidx):
        rp_dict = {}
        assert set(rp_sym_dict.keys()) == set(self.cell_lines)
        for key, value in rp_sym_dict.items():
            rp_dict[key] = _get_rpert_from_cpx(
                cpx, value, self.nodes, self.rglob[key].columns, solidx=solidx
            )
        return rp_dict

    # def _gen_deviations_dict(self, cpx, solidx):
    #     dev_dict = {}
    #     for cl in self._cell_lines:
    #         dev_dict[cl] = get_deviations_from_cpx(cpx, self._nodes,
    #                                                prefix=cl,
    #                                                solidx=solidx)
    #     return dev_dict


##############################################################################
#
# Helper functions (private)


def _get_rloc_from_cpx(cpx, nodes, prefix=None, solidx=0):
    """Construct local response matrix from cplex solution.

    Parameter
    ---------
    cpx :  cplex.Cplex object

    nodes : Tuple of nodes in the reconstructed network

    prefix : str. (optional)
        Used if rloc is part of cell line panel

    solidx : int. (optional)
        Used to select solution from solution pool

    Returns
    -------
    Square pandas.DataFrame
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

    df = pd.DataFrame(mat)
    df.index = nodes
    df.columns = nodes

    return df


def _get_deviations_from_cpx(cpx, nodes, prefix=None, solidx=0):
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


def _get_rpert_from_cpx(cpx, rp_sym, nodes, colnames, solidx=0):
    """Construct perturbation matrix from cplex solution.

    Parameters
    ----------
    cpx : cplex.Cplex object with feasible solution

    rp_sym :  matrix containing (names of) the perturbations form the
                perturbations experiment

    Returns
    -------
    matrix with the reconstructed strengths of the perturbations
    """
    assert cpx.solution.is_primal_feasible()

    allvars = cpx.variables.get_names()
    vars_lst = [var for var in allvars if var.startswith('rp_')]
    vars_vals = cpx.solution.pool.get_values(solidx, vars_lst)
    vars_dict = dict(zip(vars_lst, vars_vals))

    nn, npert = np.shape(rp_sym)

    mat = np.array([0.] * nn * npert).reshape(nn, npert)

    for ix, jx in zip(*np.nonzero(rp_sym)):
        for symbol in rp_sym[ix][jx].free_symbols:
            mat[ix][jx] += vars_dict[str(symbol)]

    df = pd.DataFrame(mat)
    df.index = nodes
    df.columns = colnames

    return df

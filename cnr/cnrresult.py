"""Classes and functions to store and analyze CNR optimization results."""

import re
import sympy
import numpy as np
import pandas as pd
import cnr.cplexutils
import cnr.maxlikelysol
from cnr.data import PerturbationPanel
import cnr.cnrutils


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

    Attributes
    ----------
    vardict : dict

    objective_value : float

    imap : binary pandas.DataFrame
        Indicates presence/absence of interaction

    rloc : dict of pandas.DataFrames

    rpert :  dict of pandas.DataFrames

    rpert_sym : dict of pandas.DataFrames of dtype sympy.Symbol

    rglob_predicted : dict of pandas.DataFrames
        obtained by computing rloc^-1 * rpert

    bounds : dict
        Bounds on parameter values.

    Methods
    -------
    add_maxlikelysol
    """
    # pylint: disable=too-many-instance-attributes
    # Eight is reasonable in this case.

    def __init__(self, p, solidx=0):
        """Initialilize object."""
        PerturbationPanel.__init__(
            self, nodes=p.nodes, perts=p.perts, pert_annot=p.pert_annot,
            ds_acting_perts=p.ds_acting_perts, rglob=p.rglob)

        self.vardict = _get_vardict_from_cpx(p.cpx, solidx)
        self.objective_value = p.cpx.solution.pool.get_objective_value(solidx)
        self.imap = _get_imap_from_cpx(p.cpx, self._nodes, solidx)
        self.rloc = self._gen_rloc_dict(p.cpx, solidx)
        self.rpert = self._gen_rpert_dict(p.cpx, p.rpert_dict, solidx=solidx)
        self.rpert_sym = p.rpert_dict
        self.bounds = _get_bounds_from_cpx(p.cpx)
        self.residuals = {
            cl: _get_residuals_from_cpx(
                p.cpx, self.nodes, self.rglob[cl].columns, prefix=cl,
                solidx=solidx
            ) for cl in self.cell_lines
        }
        self.meta_info = _extract_meta_info(p)

    @property
    def prediction_error(self):
        """"Calculate difference between observed and measured rglob."""
        mats = cnr.cnrutils.error(self.rglob_predicted, self.rglob, sum=False)
        return {cl: pd.DataFrame(mat, self.nodes, self.rglob[cl].columns) for
                cl, mat in mats.items()}

    @property
    def prediction_error_total(self):
        """"Calculate difference between observed and measured rglob."""
        return cnr.cnrutils.error(self.rglob_predicted, self.rglob,
                                  sum=True)

    @property
    def allowed_deviations(self):
        """Return indicator variables relating to difference between lines."""
        return {var: self.vardict[var] for var in self.vardict.keys()
                if (var.startswith('IDev_') or
                    var.startswith('IrpDev_') or
                    var.startswith('IrpDSDev_'))}

    @property
    def mssr(self):
        """Return mean sum of squares of residuals."""
        mean_ssr = 0
        for cl in self.cell_lines:
            n_res = np.size(self.residuals[cl])
            mean_ssr += np.sum(np.array(np.square(self.residuals[cl]))) / n_res
        mean_ssr = mean_ssr / len(self.cell_lines)
        return mean_ssr

    @property
    def rglob_predicted(self):
        """Return predicted global response matrix."""
        return cnr.cnrutils.predict_response(self.rloc, self.rpert)

    def add_maxlikelysol(self, method=None, options=None):
        """Add  MaxLikelySol instance.

        Parameters
        ----------
        method : (optional)

        options : (optional)
        """
        self.maxlikelysol = cnr.maxlikelysol.MaxLikelySol(
            self, method=method, options=options)

    def print_meta_info(self):
        """Print parameter settings used in network reconstruction."""
        for par, val in self.meta_info:
            print(par + ': ' + str(val))

    def deviations_overview(self):
        """Summarize non-zero deviations.

        Returns
        -------
        pd.DataFrame
        """
        indicators = [key for key, val in
                      self.allowed_deviations.items() if val == 1]
        df = pd.DataFrame(columns=list(self.cell_lines) + ['mean'],
                          index=indicators)
        names = []

        for i in indicators:
            info = i.split('_')
            assert info[0] in ['IDev', 'IrpDev', 'IrpDSDev'], str(
                i) + ' has unexpected form'
            if info[0] == 'IDev':
                vars_lst = ['_'.join(['r', cl] + info[1:]) for cl in
                            self.cell_lines]
                vars_vals = [self.vardict[v] for v in vars_lst]
                vars_vals.append(np.mean(vars_vals))
                df.ix[i] = vars_vals
                names += [('_'.join(['r'] + info[1:]))]
            elif info[0] == 'IrpDev':
                vars_lst = ['_'.join(['rp', cl] + info[1:]) for
                            cl in self.cell_lines]
                vars_vals = [self.vardict[v] for v in vars_lst]
                vars_vals.append(np.mean(vars_vals))
                df.ix[i] = vars_vals
                names.append('_'.join(['rp'] + info[1:]))
            elif info[0] == 'IrpDSDev':
                vars_lst = ['_'.join(['rpDS', cl] + info[1:]) for
                            cl in self.cell_lines]
                vars_vals = [self.vardict[v] for v in vars_lst]
                vars_vals.append(np.mean(vars_vals))
                df.ix[i] = vars_vals
                names.append('_'.join(['rpDS'] + info[1:]))
        df.index = names
        return df.sort_index()

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

    def _gen_deviations_dict(self, cpx, solidx):
        dev_dict = {}
        for cline in self._cell_lines:
            dev_dict[cline] = _get_deviations_from_cpx(cpx, self._nodes,
                                                       prefix=cline,
                                                       solidx=solidx)
        return dev_dict


class CnrResultPool(PerturbationPanel):
    """Hold information of panel network reconstruction solution pools.

    To solve CnrResult object p, run:
        p.cpx.solve()
        p.cpx.populate_solution_pool()

    Parameters
    ----------
    p : CnrProblem instance

    Attributes
    ----------
    solutions : dict of CnrResult
        holds the actual model solutions

    objective_values : dict
        keys are solution ids

    meta_info : dict
        parameters of CNR optimization used

    Methods
    -------
    add_maxlikelysols()

    filter_solutions(cutoff = 1.1)
    """

    def __init__(self, p):
        """Initialize with solved CnrProblem object."""
        assert p.cpx.solution.is_primal_feasible(), """p has no solution. Try
            p.cpx.solve() and p.populate_solution_pool()"""
        PerturbationPanel.__init__(
            self, nodes=p.nodes, perts=p.perts, pert_annot=p.pert_annot,
            ds_acting_perts=p.ds_acting_perts, rglob=p.rglob)
        self.solutions = {solidx: CnrResult(p, solidx=solidx) for solidx in
                          range(p.cpx.solution.pool.get_num())}
        self.meta_info = _extract_meta_info(p)

    @property
    def nsols(self):
        """Number of solutions currently in solution pool."""
        return len(self.solutions)

    @property
    def objective_values(self):
        """Dict with solution IDs as keys and objective values as values."""
        return {solidx: sol.objective_value for solidx, sol in
                self.solutions.items()}

    def add_maxlikelysols(self):
        """Add solution that minimizes the error."""
        # ids = self.solutions.keys()
        print('Finding min error solution for ' +
              str(self.nsols) + ' solutions:')
        counter = 0
        for solidx, sol in self.solutions.items():
            counter += 1
            print('Optimizing solution id: ' + str(solidx) +
                  ', index: ' + str(counter))
            sol.add_maxlikelysol()

    def filter_solutions(self, cutoff=1.1):
        """Remove solution with an objective value > cutoff * best_obj from dict.

        Parameters:
        ----------
        cutoff : float > 1
            Default = 1.1
        """
        assert cutoff > 1, 'Cutoff should be > 1'
        best_objective_value = self.objective_values[0]
        nsols_pre = self.nsols
        removed = 0
        for solidx, objective_val in self.objective_values.items():
            if objective_val > cutoff * best_objective_value:
                removed += 1
                del self.solutions[solidx]
        nsols_post = self.nsols
        assert nsols_post + removed == nsols_pre
        print(str(removed) + ' of ' + str(nsols_pre) +
              ' solutions removed from solution pool.')

##############################################################################
#
# Helper functions (private)

# For extracting information from cplex object ------------------------------
def _get_vardict_from_cpx(cpx, solidx):
    """Retun dict with var: value as entries, for selected solution."""
    var_names = cpx.variables.get_names()
    var_vals = cpx.solution.pool.get_values(solidx, var_names)
    return dict(zip(var_names, var_vals))


def _get_bounds_from_cpx(cpx):
    tmp = zip(cpx.variables.get_names(),
              cpx.variables.get_lower_bounds(),
              cpx.variables.get_upper_bounds())

    bounds = dict()
    for name, lower, upper in tmp:
        bounds[name] = (lower, upper)
    return bounds


def _get_imap_from_cpx(cpx, nodes, solidx=0):
    """
    Returns: interaction map, square np.array containing 0s and 1s.

    0: no interaction
    1: interaction
    """
    assert cpx.solution.is_primal_feasible()
    assert solidx < cpx.solution.pool.get_num()

    allvars = cpx.variables.get_names()
    vars_lst = [var for var in allvars if var.startswith('I_')]
    vars_vals = cpx.solution.pool.get_values(solidx, vars_lst)
    vars_dict = dict(zip(vars_lst, vars_vals))

    mat = np.array([0] * len(nodes)**2).reshape(len(nodes), len(nodes))
    reg_sp = re.compile('(' + '|'.join(nodes) + ')_(' + '|'.join(nodes) + ')$')
    for key, value in vars_dict.items():
        # Extract nodes from key
        sps = re.findall(reg_sp, key)
        assert len(sps) == 1, key + " has unexpected form"
        sps = sps[0]
        assert len(sps) == 2, key + " has unexpected form"
        i = nodes.index(sps[0])
        j = nodes.index(sps[1])
        assert '_'.join(['I', sps[0], sps[1]]) in vars_lst
        mat[i][j] = value

    df = pd.DataFrame(mat)
    df.index = nodes
    df.columns = nodes

    return df


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

    # pylint: disable=too-many-instance-attributes
    # Eight is reasonable in this case.

    base = "r"
    if prefix:
        assert isinstance(prefix, str)
        base = "_".join([base, prefix])
    assert cpx.solution.is_primal_feasible()
    assert solidx < cpx.solution.pool.get_num()

    allvars = cpx.variables.get_names()
    vars_lst = [var for var in allvars if var.startswith(base + '_')]
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
        mat[nodes.index(sps[0])][nodes.index(sps[1])] = value

    df = pd.DataFrame(mat)
    df.index = nodes
    df.columns = nodes

    return df


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
    vars_lst = [var for var in allvars if var.startswith('rp')]
    vars_vals = cpx.solution.pool.get_values(solidx, vars_lst)
    vars_dict = dict(zip(vars_lst, vars_vals))

    nn, npert = np.shape(rp_sym)

    mat = np.array([0.] * nn * npert).reshape(nn, npert)

    for ix, jx in zip(*np.nonzero(rp_sym)):
        for symbol in rp_sym[ix][jx].free_symbols:
            # print(symbol)
            mat[ix][jx] += vars_dict[str(symbol)]

    df = pd.DataFrame(mat)
    df.index = nodes
    df.columns = colnames

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


def _get_residuals_from_cpx(cpx, nodes, perts, prefix=None, solidx=0):
    """Return data-frame containing residuals.

    Parameters:
    ----------
    cpx : cplex.Cplex object with feasible solution

    nodes :  Tuple of nodes in the reconstructed network

    perturbations : list
        Perturbation names

    prefix: str. (optional)
        Used if rloc is part of cell line panel

    solidx : int. (optional)
        Used to select solution from solution pool
    """
    base = "res"
    if prefix:
        assert isinstance(prefix, str)
        base = "_".join([base, prefix])
    assert cpx.solution.is_primal_feasible()
    assert solidx < cpx.solution.pool.get_num()

    allvars = cpx.variables.get_names()
    vars_lst = [var for var in allvars if var.startswith(base)]
    vars_vals = cpx.solution.pool.get_values(solidx, vars_lst)
    vars_dict = dict(zip(vars_lst, vars_vals))

    nn = len(nodes)
    npars = len(vars_lst) // len(nodes)
    assert npars == len(vars_lst) / len(nodes)

    mat = np.array([0] * nn * npars, dtype=sympy.Symbol).reshape(nn, npars)
    for i in range(nn):
        for j in range(npars):
            var = '_'.join([base, nodes[i], str(j)])
            mat[i][j] = vars_dict[var]

    df = pd.DataFrame(data=mat, index=nodes, columns=perts)

    return df

def _extract_meta_info(prob):
    meta_info = dict()
    meta_info['eta'] = prob.eta
    meta_info['theta'] = prob.theta
    if prob.maxints:
        meta_info['maxints'] = prob.maxints
    if prob.maxdevs:
        meta_info['maxdevs'] = prob.maxdevs
    if prob.prior_network:
        meta_info['maxints'] = len(prob.prior_network)
    return meta_info

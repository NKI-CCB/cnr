"""Functions and classes to find maximum likelyhood solution."""

import sys

import numpy as np
import pandas as pd
import scipy

import cnr.cnrutils


class MaxLikelySol:
    """Hold result of Maxmimum likelyhood optimization.

    Parameters
    ----------
    sol : CnrResult

    Attributes
    ----------
    rloc : (dict of) pandas.DataFrame

    rpert : (dict of) pandas.DataFrame

    optimize_result : scipy.optimize.OptimizeResult
        Output of the optimization

    """

    def __init__(self, sol, method=None, options=None):
        """Calculate solution and extract results."""
        optsol = calculate_maxlikely_sol(sol, method=method, options=options)
        self.rloc = optsol['rloc']
        self.rpert = optsol['rpert']
        self.optimize_result = optsol['result']
        self._rglob = sol.rglob
        self._nodes = sol.nodes
        self._cell_lines = sol.cell_lines
        self.rglob_predicted = cnr.cnrutils.predict_response(
            self.rloc, self.rpert)
        self._allowed_deviations = sol.allowed_deviations
        self._cell_lines = sol.cell_lines

    @property
    def prediction_error(self):
        """"Calculate difference between observed and measured rglob."""
        return cnr.cnrutils.error(self.rglob_predicted, self._rglob, sum=False)

    @property
    def prediction_error_total(self):
        """"Calculate difference between observed and measured rglob."""
        return cnr.cnrutils.error(self.rglob_predicted, self._rglob, sum=True)

    def deviations_overview(self):
        """Summarize non-zero deviations.

        Returns
        -------
        pd.DataFrame
        """
        indicators = [key for key, val in self._allowed_deviations.items() if
                      val == 1]
        df = pd.DataFrame(columns=list(self._cell_lines) + ['mean'],
                          index=indicators)
        names = []

        for i in indicators:
            info = i.split('_')
            assert info[0] in ['IDev', 'IrpDev'], (str(i) +
                                                   ' has unexpected form')
            if info[0] == 'IDev':
                vars_vals = [self.rloc[cl][info[2]][info[1]]
                             for cl in self._cell_lines]
                vars_vals.append(np.mean(vars_vals))
                df.ix[i] = vars_vals
                names += [('_'.join(['r'] + info[1:]))]
            elif info[0] == 'IrpDev':
                vars_vals = [self.rpert[cl][info[1]][info[2]]
                             for cl in self._cell_lines]
                vars_vals.append(np.mean(vars_vals))
                df.ix[i] = vars_vals
                names.append('_'.join(['rp'] + info[1:]))
        df.index = names
        return df.sort_index()

##############################################################################
# Functions
# For getting solution that minimizes error


def calculate_maxlikely_sol(sol, solidx=0, method=None, options=None):
    """Calculate the rloc and rpert matrices that minimize error.

    Find the values for rloc and rpert that minimize error (i.e. the difference
    the rglob_measured and rloc^-1 * rpert), subject to the constraints on the
    network topology and allowed difference obtained by CNR optimization.

    Parameters
    ----------
    sol : CnrResult object

    solidx : int
        If a SolPanelPool object is given, solidx is used to select which
        solution from the solution pool to use.

    kwargs : valid arguments for sci

    Returns
    -------
    {
    "rloc" : dict of pd.DataFrames,
    "rpert": dict of pd.DataFrames,
    "result": scipy.minimize.OptimizeResult
    }
    """
    nodes = sol.nodes
    cell_lines = sol.cell_lines
    rglob_measured = sol.rglob
    rpert_symbols = sol.rpert_sym

    interactions_same = []
    interactions_diff = []

    perturbations_same = []
    perturbations_diff = []

    for key, val in sol.allowed_deviations.items():
        i_nodes = key.split('_')[1:]
        if key.startswith('IDev'):
            i_indicator = sol.vardict['_'.join(['I'] + i_nodes)]
            if val == 0. and i_indicator == 1:
                interactions_same.append(i_nodes)
            elif val == 1. and i_indicator == 1:
                interactions_diff.append(i_nodes)
            elif i_indicator == 0:
                pass
            else:
                sys.exit()
        elif key.startswith('IrpDev'):
            if val == 0.:
                perturbations_same.append(i_nodes)
            elif val == 1.:
                perturbations_diff.append(i_nodes)
            else:
                sys.exit()

    init = []
    varnames = []

    for i in interactions_same:
        init.append(sol.vardict['_'.join(['r', cell_lines[0]] + i)])
        varnames.append('_'.join(['r', 'MEAN'] + i))

    for i in interactions_diff:
        for cell_line in cell_lines:
            init.append(sol.vardict['_'.join(['r', cell_line] + i)])
            varnames.append('_'.join(['r', cell_line] + i))

    init_rp = []
    varnames_rp = []

    for pert in perturbations_same:
        init_rp.append(sol.vardict['_'.join(['rp', cell_lines[0]] + pert)])
        varnames_rp.append('_'.join(['rp', 'MEAN'] + pert))

    for pert in perturbations_diff:
        for cell_line in cell_lines:
            init_rp.append(
                sol.vardict['_'.join(['rp', cell_line] + pert)])
            varnames_rp.append('_'.join(['rp', cell_line] + pert))

    # Add bounds to variables
    # bounds = [(-bound, bound) for i in range(len(init + init_rp))]
    bounds = []
    for var in varnames + varnames_rp:
        varinfo = var.split('_')
        if varinfo[1] == "MEAN":
            varname = '_'.join([varinfo[0], sol.cell_lines[0],
                                varinfo[2], varinfo[3]])
        else:
            varname = var
        bounds.append(sol.bounds[varname])

    opt = scipy.optimize.minimize(
        _construct_objective_function, init + init_rp,
        args=(varnames + varnames_rp, nodes, cell_lines,
              rglob_measured, rpert_symbols),
        bounds=bounds,
        method=method,
        options=options
    )

    rvals_opt = []
    rvar_names = []

    rpvals_opt = []
    rpvar_names = []
    for var, val in zip(varnames + varnames_rp, opt.x):
        if var.split('_')[0] == 'r':
            rvals_opt.append(val)
            rvar_names.append(var)
        elif var.split('_')[0] == 'rp':
            rpvals_opt.append(val)
            rpvar_names.append(var)
        else:
            sys.exit(var + ' is invalid variable name')

    rloc_dict = _construct_rloc_matrices(
        rvals_opt, rvar_names, nodes, cell_lines
    )
    rpert_dict = _construct_rpert_matrices(
        rpvals_opt, rpvar_names, nodes, cell_lines, rpert_symbols
    )

    for cl, rloc in rloc_dict.items():
        rloc_dict[cl] = pd.DataFrame(data=rloc, index=nodes, columns=nodes)

    for cl, rpert in rpert_dict.items():
        cols = sol.rglob[cl].columns
        rpert_dict[cl] = pd.DataFrame(data=rpert, index=nodes, columns=cols)

    return {'rloc': rloc_dict, 'rpert': rpert_dict, 'result': opt}


def _construct_rloc_matrices(vals, varnames, nodes, cell_lines):
    # Construct dict empty rloc matrices
    vars_dict = dict(zip(varnames, vals))
    rloc_dict = dict()
    for cl in cell_lines:
        rloc_dict[cl] = -1 * np.identity(len(nodes), dtype=float)

    # Fill matrices with specified values
    for var, val in vars_dict.items():
        cl = var.split('_')[1]
        idx_i = nodes.index(var.split('_')[2])
        idx_j = nodes.index(var.split('_')[3])
        if cl == 'MEAN':
            for rloc in rloc_dict.values():
                rloc[idx_i][idx_j] = val
        elif cl in cell_lines:
            rloc_dict[cl][idx_i][idx_j] = val
        else:
            sys.exit(cl + ' is not a valid cell line name')

    return rloc_dict


def _construct_rpert_matrices(vals, varnames, nodes, cell_lines,
                              rpert_symbols):

    vars_dict = dict()
    for var, val in zip(varnames, vals):
        if var.split('_')[1] in cell_lines:
            vars_dict[var] = val
        elif var.split('_')[1] == 'MEAN':
            for cl in cell_lines:
                vars_dict[var.replace('MEAN', cl)] = val
        else:
            sys.exit(cl + ' is invalid cell line name')

    rp_dict = dict()
    for cl, rp_sym in rpert_symbols.items():
        nn = np.shape(rp_sym)[0]
        npert = np.shape(rp_sym)[1]

        mat = np.array([0.] * nn * npert).reshape(nn, npert)

        for ix, jx in zip(*np.nonzero(rp_sym)):

            for sym in rp_sym[ix][jx].free_symbols:
                mat[ix][jx] += vars_dict[str(sym)]

        rp_dict[cl] = mat

    return rp_dict


def _construct_objective_function(vals, varnames, nodes, cell_lines, rglobs,
                                  rpert_symbols):
    """Construct objective function for use in calculate minErrorSol."""
    rvals = []
    rvar_names = []

    rpvals = []
    rpvar_names = []
    for var, val in zip(varnames, vals):
        if var.split('_')[0] == 'r':
            rvals.append(val)
            rvar_names.append(var)
        elif var.split('_')[0] == 'rp':
            rpvals.append(val)
            rpvar_names.append(var)
        else:
            sys.exit(var + ' is invalid variable name')
    rlocs = _construct_rloc_matrices(rvals, rvar_names, nodes,
                                     cell_lines)
    rperts = _construct_rpert_matrices(
        vals=rpvals, varnames=rpvar_names, nodes=nodes, cell_lines=cell_lines,
        rpert_symbols=rpert_symbols)
    rglob_predicted = cnr.cnrutils.predict_response(rlocs, rperts)
    err = 0
    for cl in cell_lines:
        err += cnr.cnrutils.error(rglob_predicted[cl], rglobs[cl], sum=True)
    return err

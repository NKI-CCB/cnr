"""Classes and functions to generate cplex.Cplex for CNR problem."""

import cplex
import numpy as np
import sympy
import cnr.cplexutils
from cnr.data import PerturbationPanel


###############################################################################
#
# Helper functions

def _add_direct_target(nodes, pert_name, target, base_name):
    to_add = np.zeros([len(nodes), 1], dtype=sympy.Symbol)
    indx = nodes.index(target)
    to_add[indx] += sympy.Symbol('_'.join([base_name, pert_name, target]))
    return to_add


def _add_downstream_effect(nodes, pert_name, pert_target, prior_network,
                           base_name):
    to_add = np.zeros([len(nodes), 1], dtype=sympy.Symbol)
    # If restricted to prior network
    if prior_network:
        for ds_target, source in prior_network:
            if source == pert_target:
                indx = nodes.index(ds_target)
                to_add[indx] += sympy.Symbol(
                    '_'.join([base_name, pert_name, ds_target]))
    else:
        downstream_nodes = set(nodes) - set([pert_target])
        for ds_target in downstream_nodes:
            indx = nodes.index(ds_target)
            to_add[indx] += sympy.Symbol(
                '_'.join([base_name, pert_name, ds_target]))
    return to_add


def generate_rpert_symbols(nodes, perturbations, pert_annot, ds_acting_perts,
                           prior_network=None, prefix=""):
    """"Generate matrix containing perturbations as symbols.

    Parameters
    ----------
    nodes : tuple of strings

    prior_network : list of 2-tuples, optional

    Returns
    -------
    square np.array(dtype=sympy.Symbol)
        element (i, j) is the perturbation(s) applied directly to node i
        the j-th perturbation.
    """
    base = "rp"
    baseds = 'rpDS'
    if prefix:
        assert isinstance(prefix, str)
        base = '_'.join([base, prefix])
        baseds = '_'.join([baseds, prefix])

    rpert_sym = np.empty([len(nodes), 0], dtype=sympy.Symbol)

    # For each applied (combination of) perturbations
    for pert_name_lst in perturbations:
        new_pert = np.zeros([len(nodes), 1], dtype=sympy.Symbol)
        if isinstance(pert_name_lst, str):
            pert_name_lst = [pert_name_lst]

        for pert_name in pert_name_lst:
            pert_target = pert_annot[pert_name]

            # Add direct targets effects
            if pert_name not in ds_acting_perts:
                new_pert += _add_direct_target(nodes,
                                               pert_name, pert_target, base)

            # Add indirect effects
            # if pert_name in ds_acting_perts:
            else:
                # if pert_name in ds_acting_perts:
                # print("Indirect: " + pert_name + ' ' + pert_target)
                new_pert += _add_downstream_effect(
                    nodes, pert_name, pert_target, prior_network, baseds
                )
        rpert_sym = np.append(rpert_sym, new_pert, axis=1)

    return rpert_sym


def generate_rloc_symbols(nodes, prior_network, prefix=None):
    """"Generate matrix containing local response coefficients as symbols.

    Parameters
    ----------
    nodes : tuple of strings

    prior_network : list of 2-tuples, optional

    Returns
    -------
    NxN np array with dtype=sympy.Symbol with as ij-th elements  r_nodei_nodej
    and on the diagonal: r_i_i
    """
    nn = len(nodes)
    mat = np.zeros((nn, nn), dtype=sympy.Symbol)
    BASE = "r_"
    if prefix:
        assert isinstance(prefix, str)
        BASE = BASE + prefix + "_"

    if prior_network:
        for ia in prior_network:
            i = nodes.index(ia[0])
            j = nodes.index(ia[1])
            mat[i][j] = sympy.Symbol(BASE + ia[0] + '_' + ia[1])
        for i in range(nn):
            mat[i][i] = sympy.Symbol('r_i_i')
    else:
        for i in range(nn):
            for j in range(nn):
                if i == j:
                    mat[i][j] = sympy.Symbol('r_i_i')
                else:
                    mat[i][j] = sympy.Symbol(BASE + nodes[i] + '_' + nodes[j])
    return mat


def generate_indicators(nodes, prior_network, base='I'):
    """Return list with indicator names (optional: based on prior network)."""
    ind_lst = []
    if prior_network:
        for i in prior_network:
            ind_lst.append('_'.join([base, i[0], i[1]]))
    else:
        for i in nodes:
            for j in nodes:
                if i != j:
                    ind_lst.append('_'.join([base, i, j]))
    return ind_lst

###############################################################################
#
# Classes


class CnrProblem(PerturbationPanel):
    """Class to generate CNR problem and associated cplex.Cplex object.

    Inherits from PerturbationPanel. Each cell line must have same
    interactions, but is allowed to have different local response coefficient
    values.

    Parameters
    ---------
    perturbation_panel : PerturbationPanel object

    bounds : float > 0, optional (default=cplex.infinity)
        Set the bounds on all continious variables in the models. This is
        overwritten if more specific (r_bounds, rp_bounds, res_bounds,
        dev_bounds) bounds are set.

    r_bounds : float > 0, optional
        Sets/overwrites the bounds on local response coefficient

    rp_bounds : float > 0, optional
        Sets/overwrites on bounds on the perturbation strengths

    dev_bounds : float > 0, optional
        Sets/overwrites the bounds on the deviations from mean.

    res_bounds: float > 0, optional
        Sets/overwrites the bounds on the residuals. Setting these
        tight might increase stability of the problem, but can also make it
        infeasible.

    eta : float, optional (default = 0.0)
        Weigth of interactions vs errors in objective functions. Higher eta
         favors sparser solutions.

    theta : float, optional (default = 0.0)
        Weight of deviation from mean in objective function. Higher values
        favor more similar solutions.

    prior_network :  List of 2-tuples, optional
        If provided, possible interactions are restricted to this set

    maxints : int, optional
        Number of interactions restricted to 'maxints'

    maxdevs : int, optional
        Number of edges that differ between cell lines is restricted to
        'maxints'

    Attributes
    ----------
    nodes : tuple of strings
        Names of the nodes in the model

    rglob : dict of panda.DataFrame
        Measured responses. Input data for the network reconstruction.

    cell_lines : list of str
        Names of the cell lines in the panel

    perts : list of lists
        Applied perturbations.

    pert_annot : dict
        Annotations of the perturbations. Keys are applied perturbations,
        values are targeted nodes.

    cpx : cplex.Cplex object
        Contains the Cnr cplex problem formulation.

    Methods
    -------
    set_edge_sign(edge, sign)

    set_pert_sign(perturbation, sign)
    """

    def __init__(
            self, pert_panel,
            bounds=cplex.infinity,
            r_bounds=None,
            rp_bounds=None,
            dev_bounds=None,
            res_bounds=None,
            eta=0.0, theta=0.0,
            maxints=None, maxdevs=None,
            prior_network=None):
        """Initialization from PerturbationPanel object."""
        # From input
        PerturbationPanel.__init__(
            self, nodes=pert_panel.nodes, perts=pert_panel.perts,
            pert_annot=pert_panel.pert_annot,
            ds_acting_perts=pert_panel._ds_acting_perts,
            rglob=pert_panel.rglob)
        self._eta = eta
        self._theta = theta
        self._prior = prior_network
        self._maxints = maxints
        self._maxdevs = maxdevs
        self._r_bounds = r_bounds if r_bounds is not None else bounds
        self._rp_bounds = rp_bounds if rp_bounds is not None else bounds
        self._res_bounds = res_bounds if res_bounds is not None else bounds
        self._dev_bounds = dev_bounds if dev_bounds is not None else bounds
        # Construct helper objects
        self._rloc_dict = self._gen_rloc()
        self._rpert_dict = self._gen_rpert()
        self._rloc_vars = self._gen_rloc_vars()
        self._rp_vars = self._gen_rp_vars()
        self._rpds_vars = [rp for rp in self._rp_vars if rp.split("_")[
            0] == 'rpDS']
        self._indicators = generate_indicators(self.nodes, self._prior)
        self._dev_indicators = generate_indicators(self.nodes, self._prior,
                                                   base='IDev')
        self._rp_indicators = self._gen_rp_indicators()
        self._error_terms = []
        self._dev_vars = []
        self._rpdev_vars = []
        self.cpx = self.add_cpx()

    @property
    def rpert_dict(self):
        """Dict with rpert matrices, elements are sympy.Symbols."""
        return self._rpert_dict

    @property
    def eta(self):
        """Return value of eta, higher eta favors sparser solutions."""
        return self._eta

    @property
    def theta(self):
        """Return value of theta, higher values favor more similar solutoins."""
        return self._theta

    @property
    def prior_network(self):
        """Return allowed edges in the solution as list of tuples."""
        return self._prior

    @property
    def maxints(self):
        """Return maximally allowed number of edges in solution."""
        return self._maxints

    @property
    def maxdevs(self):
        """Return maximally allowed differing edges or perturbations."""
        return self._maxdevs

    def _gen_rloc(self):
        rldict = dict()
        for name in self.cell_lines:
            rldict[name] = generate_rloc_symbols(
                self.nodes, self._prior, name)
        return rldict

    def _gen_rpert(self):
        rpdict = dict()

        for name in self.cell_lines:
            rpdict[name] = generate_rpert_symbols(
                self.nodes, self.perts, self.pert_annot,
                self._ds_acting_perts, prefix=name, prior_network=self._prior
            )
        return rpdict

    def _gen_rloc_vars(self):
        """Generate list of local response coefficient names.

        Returns list of strings
        """
        rnames = set()
        for rloc in self._rloc_dict.values():
            rnames.update(set(rloc.flatten()))
        rnames = [str(r) for r in list(rnames)]
        rnames = sorted(rnames)
        assert 'r_i_i' in rnames
        rnames.remove('r_i_i')
        if '0' in rnames:
            rnames.remove('0')
        return rnames

    def _gen_rp_vars(self):
        rpnames = set()
        for rpert in self._rpert_dict.values():
            for rp in set(rpert[np.nonzero(rpert)]):
                rpnames.update(rp.free_symbols)
        rpnames = [str(r) for r in list(rpnames)]
        rpnames = sorted(rpnames)
        return rpnames

    def _gen_rp_indicators(self):
        indicator_list = []
        for rpvar in self._rp_vars:
            rpvar_elements = rpvar.split('_')
            # rpvar is expected to have the form:
            # rp_cellline_pertname_node or rpDS_cellline_pertname_node
            assert rpvar_elements[0] in {'rp', "rpDS"}, (rpvar + ' has \
                unexpected form')
            assert rpvar_elements[1] in self.cell_lines
            assert rpvar_elements[2] in self.pert_annot.keys()
            assert rpvar_elements[3] in self.nodes

            if rpvar_elements[0] == "rp":
                indicator_list.append(
                    'IrpDev_' + '_'.join(rpvar.split('_')[2:]))
            else:
                indicator_list.append(
                    'IrpDSDev_' + '_'.join(rpvar.split('_')[2:]))
        return list(set(indicator_list))

    def merge_indicators(self, to_merge, merged_indicator_name):
        """ONLY WORKS FOR PERTURBATIONS."""
        indicators_to_merge = []
        # Find names of indicators to use.
        for ind in self._rp_indicators:
            pert = ind.split('_')[1]
            if pert in to_merge:
                ind = indicators_to_merge.append(ind)
        assert len(indicators_to_merge) == len(to_merge)

        self.cpx.variables.add(names=[merged_indicator_name],
                               types=[self.cpx.variables.type.binary],
                               lb=[0], ub=[1], obj=[self._theta])

        # Change objective coefficients of merged indicators
        ind_seq = [(var, 0) for var in indicators_to_merge]
        self.cpx.objective.set_linear(ind_seq)

        #  Change  max_deviation_from_mean if needed
        if self._maxdevs is not None:
            # Reomve old constraint
            self.cpx.linear_constraints.delete('max_deviations_from_mean')
            use_indicators = set(self._dev_indicators + self._rp_indicators +
                                 [merged_indicator_name]) - \
                set(indicators_to_merge)
            use_indicators = list(use_indicators)
            constr = cplex.SparsePair(
                use_indicators, [1] * len(use_indicators))

            self.cpx.linear_constraints.add(lin_expr=[constr],
                                            rhs=[self._maxdevs], senses=["L"],
                                            names=["max_deviations_from_mean"])

        # Set indicator relation
        equal_ind_constraints = []
        equal_ind_names = []
        for ind in indicators_to_merge:
            equal_ind_constraints.append(
                cplex.SparsePair([merged_indicator_name, ind], [1, -1])
            )
            equal_ind_names.append('merge' + ind)
        self.cpx.linear_constraints.add(
            lin_expr=equal_ind_constraints, rhs=[0] * len(indicators_to_merge),
            senses=['E'] * len(indicators_to_merge), names=equal_ind_names
        )

    def set_edge_sign(self, edge, sign):
        """Restrict edge to be positive or negative.

        Parameters
        ----------
        edge : tuple or list of tuples.

        sign : {'pos', 'neg'}
        """
        varnames = []
        if isinstance(edge, tuple):
            edge = [edge]

        for e in edge:
            for cln in self.cell_lines:
                varnames.append('_'.join(['r', cln, e[0], e[1]]))

        if sign == 'pos':
            cnr.cplexutils.set_vars_positive(self.cpx, varnames)
        elif sign == 'neg':
            cnr.cplexutils.set_vars_negative(self.cpx, varnames)
        else:
            raise ValueError("sign should be one of: ('pos', 'neg')")

    def set_pert_sign(self, pert, sign):
        """Restrict edge to be positive or negative.

        Input:
        pert: tuple or list of tuples
            Tuple should be (perturbation name, perturbation target), e.g.
            ('plx', 'MEK')
        sign: string, 'pos' or 'neg'
        """
        varnames = []
        if isinstance(pert, tuple):
            pert = [pert]

        for p in pert:
            assert p[0] in self.pert_annot.keys()
            assert p[1] in self.nodes
            for cln in self.cell_lines:
                if p[0] in self._ds_acting_perts:
                    varnames.append('_'.join(['rpDS', cln, p[0], p[1]]))
                else:
                    varnames.append('_'.join(['rp', cln, p[0], p[1]]))

        if sign == 'pos':
            cnr.cplexutils.set_vars_positive(self.cpx, varnames)
        elif sign == 'neg':
            cnr.cplexutils.set_vars_negative(self.cpx, varnames)
        else:
            raise ValueError("sign should be one of: ('pos', 'neg')")

    def set_interactions_status(self, interaction_list, status):
        """Force interaction to be absent/persent.

        Parameters:
        -----------
        interaction_list: list of tuples
            tuples should have form (node_i, node_j): with n_j --> n_i

        status: {1, 0}
            1. Interaction is present, 0 absent
        """
        indicator_lst = ['_'.join(['I', n_i, n_j])
                         for n_i, n_j in interaction_list]

        for indicator in indicator_lst:
            print("setting indicator " + indicator + " to " + str(status))
            cnr.cplexutils.set_indicator_status(self.cpx, indicator, status)

    def initialize_from_solution(self, cnr_res, solidx=0):
        """Start optimization from earlier obtained solution.

        param:
        cnr_res: cnr.CnrResult
            Must be derived from same CnrPanel object.
        """
        effort_level = self.cpx.MIP_starts.effort_level.auto
        assert set(cnr_res.vardict.keys()) == set(
            self.cpx.variables.get_names())
        variables, values = zip(*cnr_res.vardict.items())
        self.cpx.MIP_starts.add(
            [list(variables), list(values)],
            effort_level
        )

    def add_cpx(self):
        """Create cplex MIQP problem."""
        # Initialize the Cplex object.
        cpx = cplex.Cplex()

        cpx.set_problem_type(cplex.Cplex.problem_type.MIQP)
        cpx.objective.set_sense(cpx.objective.sense.minimize)

        # ---------------------------------------------------------------------
        # Construct variables
        #

        # Add local response coefficients as variables
        n_rloc = len(self._rloc_vars)
        cpx.variables.add(names=self._rloc_vars,
                          types=[cpx.variables.type.continuous] * n_rloc,
                          lb=[-self._r_bounds] * n_rloc,
                          ub=[self._r_bounds] * n_rloc)

        # Add diagonal element of r matrix
        cpx.variables.add(names=['r_i_i'],
                          types=[cpx.variables.type.continuous],
                          lb=[-1.], ub=[-1.])

        # Add perturbations as variables
        n_rp = len(self._rp_vars)
        cpx.variables.add(names=self._rp_vars,
                          types=[cpx.variables.type.continuous] * n_rp,
                          lb=[-self._rp_bounds] * n_rp,
                          ub=[self._rp_bounds] * n_rp)

        # Add indicator constraint as variables
        Ni = len(self._indicators)
        cpx.variables.add(names=self._indicators,
                          types=[cpx.variables.type.binary] * Ni,
                          lb=[0] * Ni,
                          ub=[1] * Ni,
                          obj=[self._eta] * Ni)

        # Add indicator constraint as variables
        Nidev = len(self._dev_indicators)
        cpx.variables.add(names=self._dev_indicators,
                          types=[cpx.variables.type.binary] * Nidev,
                          lb=[0] * Nidev,
                          ub=[1] * Nidev,
                          obj=[self._theta] * Nidev)

        Nirpdev = len(self._rp_indicators)
        cpx.variables.add(names=self._rp_indicators,
                          types=[cpx.variables.type.binary] * Nirpdev,
                          lb=[0] * Nirpdev,
                          ub=[1] * Nirpdev,
                          obj=[self._theta] * Nirpdev)

        # ---------------------------------------------------------------------
        # Construct linear constraints

        linexps = []
        linexp_names = []

        # Loop over all cell lines in the panel

        for cell_line in self.cell_lines:
            rglob = np.array(self.rglob[cell_line])
            rloc = np.array(self._rloc_dict[cell_line])
            rpert = np.array(self._rpert_dict[cell_line])

            nn, nperts = np.shape(rglob)
            assert np.shape(rglob) == np.shape(rpert)
            assert nn == len(self.nodes)

            for node in range(nn):
                for pert in range(nperts):
                    # The coefficients of r
                    # Row i of rloc
                    var = [str(r) for r in rloc[node, :]]

                    # Column j of rglob
                    coef = list(np.array(rglob)[:, pert])

                    # Remove absent interactions
                    indices = [i for i, x in enumerate(var) if x == '0']
                    var = [i for j, i in enumerate(var) if j not in indices]
                    coef = [i for j, i in enumerate(coef) if j not in indices]

                    # Add direct perturbation if applicable
                    try:
                        for rpvar in rpert[node, pert].free_symbols:
                            var.append(str(rpvar))
                            coef.append(1.)
                    except:
                        if rpert[node, pert] == 0:
                            pass
                        else:
                            raise ValueError

                    # Add error term (also add variable)
                    errvar = '_'.join(['res', cell_line,
                                       str(self._nodes[node]),
                                       str(pert)])
                    var.append(errvar)
                    coef.append(1.)
                    self._error_terms.append(errvar)
                    cpx.variables.add(names=[errvar],
                                      types=[cpx.variables.type.continuous],
                                      lb=[-self._res_bounds],
                                      ub=[self._res_bounds])

                    linexps.append([var, coef])  # Add constraint to linexps
                    eqname = '_'.join(['eq', cell_line, str(node), str(pert)])
                    linexp_names.append(eqname)

        # Add the constraints
        cpx.linear_constraints.add(lin_expr=linexps,
                                   senses=["E"] * len(linexps),
                                   rhs=[0.] * len(linexps),
                                   names=linexp_names)

        # If set, add constraint on number of interaction
        if self._maxints:
            constr = cplex.SparsePair(self._indicators,
                                      [1] * len(self._indicators))
            cpx.linear_constraints.add(lin_expr=[constr],
                                       rhs=[self._maxints], senses=["L"],
                                       names=["max_interactions"])

        # If set, add constraint on number of differences between cell lines.
        if self._maxdevs is not None:
            all_dev_inds = self._dev_indicators + self._rp_indicators
            constr = cplex.SparsePair(all_dev_inds,
                                      [1] * len(all_dev_inds))
            cpx.linear_constraints.add(lin_expr=[constr],
                                       rhs=[self._maxdevs], senses=["L"],
                                       names=["max_deviations_from_mean"])

        # ---------------------------------------------------------------------
        # Add quadratic part to objective function
        for err in self._error_terms:
            cpx.objective.set_quadratic_coefficients(
                err, err, 1.)  # - self._eta)

        # ---------------------------------------------------------------------
        # Construct deviation from mean constraints

        # Iterate over all cell lines
        n_cell_lines = len(self.cell_lines)
        dev_constr = []
        dev_constr_names = []
        rpdev_constr = []
        rpdev_constr_names = []

        for cell_line in self.cell_lines:
            other_cell_lines = set(self.cell_lines)
            other_cell_lines.remove(cell_line)

            # Get all local response coefficients
            rloc_set = set(self._rloc_dict[cell_line].flatten()) - set('')
            rloc_lst = [str(r) for r in rloc_set]

            if '0' in rloc_lst:
                rloc_lst.remove('0')
            rloc_lst.remove('r_i_i')

            for r in rloc_lst:
                var = ["dev_" + r, r]
                # NOTE: must be floats
                coef = [-1., (n_cell_lines - 1.) / n_cell_lines]
                self._dev_vars.append("dev_" + r)
                # Add all other cell lines
                for other in other_cell_lines:
                    var.append(r.replace(cell_line, other))
                    coef.append(-1. / n_cell_lines)
                dev_constr.append([var, coef])
                dev_constr_names.append("diffeq_" + r)

            # Get all perturbations matrix entries
            rp_set = set(self._rpert_dict[cell_line].flatten())
            rp_symbol_lst = list(rp_set - {0})
            rp_lst = []
            # Split expressions and keep symbols
            for rp in rp_symbol_lst:
                rp_lst += [str(r) for r in rp.atoms()]

            for rp in rp_lst:
                var = ["rpdev_" + rp, rp]
                coef = [-1., (n_cell_lines - 1.) / n_cell_lines]
                self._rpdev_vars.append(var[0])
                for other in other_cell_lines:
                    var.append(rp.replace(cell_line, other))
                    coef.append(-1. / n_cell_lines)
                rpdev_constr.append([var, coef])
                rpdev_constr_names.append("rpdiffeq_" + rp)

        # Add the deviations from mean as variables to cpx object
        both_vars = self._dev_vars + self._rpdev_vars
        cpx.variables.add(names=both_vars,
                          types=[cpx.variables.type.continuous] *
                          len(both_vars),
                          lb=[-self._dev_bounds] * len(both_vars),
                          ub=[self._dev_bounds] * len(both_vars))

        # Add the constraints for deviations from mean
        both_constr = dev_constr + rpdev_constr
        both_constr_names = dev_constr_names + rpdev_constr_names
        cpx.linear_constraints.add(lin_expr=both_constr,
                                   senses=["E"] * len(both_constr),
                                   rhs=[0.] * len(both_constr),
                                   names=both_constr_names)

        # ---------------------------------------------------------------------
        # Construct indicator constraints for presence of edge
        for rvar in self._rloc_vars:
            # rvar is expected to have form r_cellline_nodei_nodej
            r_elements = rvar.split('_')
            assert len(r_elements) == 4
            assert r_elements[0] == 'r'
            assert r_elements[1] in self.cell_lines
            assert r_elements[2] in self.nodes and r_elements[3] in self.nodes
            ivar = '_'.join(["I", r_elements[2], r_elements[3]])
            assert ivar in self._indicators
            name = '_'.join(["Ind", r_elements[1], r_elements[2],
                             r_elements[3]])
            constr = cplex.SparsePair(ind=[rvar], val=[1.])
            cpx.indicator_constraints.add(indvar=ivar,
                                          complemented=1,
                                          rhs=0., sense='E',
                                          lin_expr=constr,
                                          name=name)

        # ---------------------------------------------------------------------
        # Construct indicator constraints for presence dowstream perturbation
        # effects
        for rpdsvar in self._rpds_vars:
            # rpDSvar is expected to have form rpDS_cellline_pert_nodej
            rpds_elements = rpdsvar.split("_")
            assert len(rpds_elements) == 4
            assert rpds_elements[0] == 'rpDS'
            assert rpds_elements[1] in self.cell_lines
            assert rpds_elements[2] in self.pert_annot.keys()
            assert rpds_elements[3] in self.nodes
            ivar = '_'.join(['I', rpds_elements[3],
                             self.pert_annot[rpds_elements[2]]])
            assert ivar in self._indicators
            name = '_'.join(['Irpsd'] + rpds_elements[1:])
            constr = cplex.SparsePair(ind=[rpdsvar], val=[1.])
            cpx.indicator_constraints.add(indvar=ivar,
                                          complemented=1,
                                          rhs=0., sense='E',
                                          lin_expr=constr,
                                          name=name)

        # ---------------------------------------------------------------------
        # Construct indicator constraints for deviation of edge from mean
        for dev in self._dev_vars:
            # dev is expected to have form
            # dev_r_cellline_nodei_nodej
            dev_elements = dev.split('_')[2:]
            assert len(dev_elements) == 3, dev + " has unexpected form"
            ivar = '_'.join(["IDev", dev_elements[1], dev_elements[2]])
            assert ivar in self._dev_indicators, " Unexpected variable in \
            construction of indicator constraints"
            name = '_'.join(["IndDev", dev_elements[0], dev_elements[1],
                             dev_elements[2]])
            constr = cplex.SparsePair(ind=[dev], val=[1.])
            cpx.indicator_constraints.add(indvar=ivar, complemented=1,
                                          rhs=0., sense='E',
                                          lin_expr=constr,
                                          name=name)
            # This indicator can only be active if the corresponding edge is
            # present. Implement as IDev - I < 1
            ivar_edge = '_'.join(["I", dev_elements[1], dev_elements[2]])
            constr = cplex.SparsePair(ind=[ivar, ivar_edge], val=[1., -1])
            cpx.linear_constraints.add(
                lin_expr=[constr], senses=["L"], rhs=[0])

        # ---------------------------------------------------------------------
        # Construct indicator constraints for deviation of perturbation from
        # mean
        for rpdev in self._rpdev_vars:
            # rpdev is expected to have form:
            # rpdev_rp_cellline_perturbation_targetnode or
            # rpdev_rpDS_cellline_perturbation_targetnode
            dev_type = rpdev.split('_')[1]
            dev_elements = rpdev.split('_')[2:]

            # todo disallow spaces in names
            # will give problems when using cplex.write(), which subsititutes
            # spaces with underscores

            assert len(dev_elements) == 3, rpdev + " has unexpected form"
            assert dev_type in {'rp', 'rpDS'}, rpdev + " has unexpected form"
            if dev_type == 'rp':
                ivar = '_'.join(['IrpDev'] + dev_elements[1:])
            elif dev_type == 'rpDS':
                ivar = '_'.join(['IrpDSDev'] + dev_elements[1:])
            assert ivar in self._rp_indicators, ivar + " has unexpected form"
            name = '_'.join(['IndRpDev'] + dev_elements)
            constr = cplex.SparsePair(ind=[rpdev], val=[1.])
            cpx.indicator_constraints.add(indvar=ivar, complemented=1,
                                          rhs=0., sense='E',
                                          lin_expr=constr,
                                          name=name)

        return cpx

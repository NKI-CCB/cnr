import cnr
import pandas as pd

PATH = "/Users/e.bosdriesz/Dropbox/projects/cnr/data/PTPN11_persistor/"
vaco_wt = pd.read_csv(PATH + 'vaco_wt_lfc.tsv', sep='\t', index_col=0)
vaco_wt = vaco_wt.drop('Blank', axis=1)

nodes = list(vaco_wt.index)
REF_NODES = ['EGFR',  # RTKs
             'MEK', 'ERK', 'P90RSK', 'GSK3AB', 'RPS6',  # MAPK
             'PI3K', 'AKT', 'MTOR',  # AKT
             'JNK', 'CJUN',
             'P38', 'IKBA',
             'P53', 'SMAD2']

vaco_wt = vaco_wt.ix[REF_NODES]
vaco_wt.columns = [c.lower() for c in vaco_wt.columns]

vaco_ko = pd.read_csv(PATH + 'vaco_ko_lfc.tsv', sep='\t', index_col=0)
vaco_ko = vaco_ko.ix[REF_NODES].drop('Blank', axis=1)
vaco_ko.columns = [c.lower() for c in vaco_ko.columns]

vaco_pe = pd.read_csv(PATH + 'vaco_pe_lfc.tsv', sep='\t', index_col=0)
vaco_pe = vaco_pe.ix[REF_NODES].drop('Blank', axis=1)
vaco_pe.columns = [c.lower() for c in vaco_pe.columns]

ALLOWED_INTERACTIONS = [
    ('MEK', 'EGFR'), ('PI3K', 'EGFR'), ('AKT', 'EGFR'),  # EGFR
    ('ERK', 'MEK'),  # MEK
    ('MEK', 'ERK'), ('EGFR', 'ERK'), ('P90RSK', 'ERK'), ('RPS6', 'ERK'),  # ERK
    ('GSK3AB', 'P90RSK'),  # P90RSK
    # GSK3AB
    ('PI3K', 'RPS6'), ('MTOR', 'RPS6'),  # RPS6
    ('AKT', 'PI3K'), ('JNK', 'PI3K'), ('P38', 'PI3K'), ('IKBA', 'PI3K'),   # PI3K
    ('MTOR', 'AKT'), ('GSK3AB', 'AKT'),  # AKT
    ('RPS6', 'MTOR'),  # ('AKT', 'MTOR'),  # MTOR
    ('CJUN', 'JNK'),  # JNK
    # CJUN
    # P38
    # IKBA
    # P53
    # SMAD2
]

TARGETS = {
    'egf': ['EGFR'],
    'hgf': ['PI3K', 'AKT', 'MEK'],
    'nrg1': ['PI3K', 'AKT', 'MEK'],
    'plx': ['MEK'],
    'mek': ['ERK'],
    'erk': ['MEK', 'EGFR', 'P90RSK', 'RPS6'],
    'pi3k': ['AKT', 'JNK', 'P38', 'IKBA'],
    'akt': ['MTOR', 'GSK3AB']
}

perts = [c.replace(' ', '').split('+') for c in vaco_wt.columns]

pp = cnr.PerturbationPanel(REF_NODES, perts, TARGETS)
pp.add_cell_line('wt', vaco_ko)
pp.add_cell_line('ko', vaco_ko)
pp.add_cell_line('pe', vaco_pe)

cnr_problem = cnr.CnrProblem(pp, prior_network=ALLOWED_INTERACTIONS)

POSITIVE_PERTURBATIONS = [
    ('egf', 'EGFR'),
    ('hgf', 'PI3K'), ('hgf', 'AKT'), ('hgf', 'MEK'),
    ('nrg1', 'PI3K'), ('nrg1', 'AKT'), ("nrg1", 'MEK'),
    ('erk', 'MEK'), ('erk', 'EGFR')
]

NEGATIVE_PERTURBATIONS = [
    ('mek', 'ERK'),
    ('erk', 'P90RSK'), ("erk", 'RPS6'),
    ('pi3k', 'AKT'), ("pi3k", 'JNK'), ('pi3k', 'P38'), ("pi3k", 'IKBA'),
    ('akt', 'MTOR'), ('akt', 'GSK3AB')
]


POSITIVE_INTERACTIONS = [
    ('MEK', 'EGFR'), ('PI3K', 'EGFR'), ('AKT', 'EGFR'),  # EGFR
    ('ERK', 'MEK'),  # MEK
    ('P90RSK', 'ERK'), ('RPS6', 'ERK'),  # ERK
    ('GSK3AB', 'P90RSK'),  # P90RSK
    # GSK3AB
    ('MTOR', 'RPS6'),  # RPS6
    ('AKT', 'PI3K'), ('JNK', 'PI3K'), ('P38', 'PI3K'), ('IKBA', 'PI3K'),  # PI3K
    ('MTOR', 'AKT'), ('GSK3AB', 'AKT'),  # AKT
    ('RPS6', 'MTOR'),  # ('AKT', 'MTOR'),  # MTOR
    ('CJUN', 'JNK'),  # JNK
    # CJUN
    # P38
    # IKBA
    # P53
    # SMAD2
]

NEGATIVE_INTERACTIONS = [
    ('MEK', 'ERK'), ('EGFR', 'ERK'),  # ERK
    ('PI3K', 'RPS6')
]

cnr_problem.set_edge_sign(POSITIVE_INTERACTIONS, 'pos')
cnr_problem.set_edge_sign(NEGATIVE_INTERACTIONS, 'neg')

cnr_problem.set_pert_sign(POSITIVE_PERTURBATIONS, 'pos')
cnr_problem.set_pert_sign(NEGATIVE_PERTURBATIONS, 'neg')

cnr_problem.cpx.solve()
cnr_sol = cnr.CnrResult(cnr_problem)

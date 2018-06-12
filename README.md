# Comparative Network Reconstruction

This repository contains the method described in the publication:
"Comparative Network Reconstruction using Mixed Integer Programming", Bosdriesz et al., 2018, https://doi.org/10.1101/243709.
Comparative Network Reconstruction (CNR) aims to reconstruct and quantify (signaling) networks from perturbation experiments, with the aim of finding relevant differences between 2 or more cell lines.

The code for the applications described in manuscript can be found in their own repository https://bitbucket.org/evertbosdriesz/cnr.

## Requirements

* Python 3.5 or 3.6
* IBM ILOG CLEX Optimization studio.
Tested for version 12.8, but it should work for version 12.6 and higher.
Free licence are available for academic use.
To use CNR, you'll need to install the python interface ([here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html) you can find instructions how)

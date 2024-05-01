# Lagrangian_GP
Accompanying sourcecode for the article

C.Offen\
*Machine learning of continuous and discrete variational ODEs with guaranteed convergence and uncertainty quantifications* (2024)\
Status: Preprint [arXiv:2404.19626](http://arxiv.org/abs/2404.19626)

[Preprint on ResearchGate](https://dx.doi.org/10.13140/RG.2.2.16171.04646),
[ArXiv author page](https://arxiv.org/a/offen_c_1.html),
[Research Information System](https://ris.uni-paderborn.de/person/85279)

To reproduce the experiments of the artcile (and more), run

`continuous/L_Learning_CertifiedGP_RUN.jl`\
`discrete/Ld_Learning_CertifiedGP_RUN.jl`\
`convergence/L_Learning_CertifiedGP_oscillator_1d_convergence_DoubleFloat.jl`

Further experiments to learn discrete Lagrangians may be viewed in\
`discrete/Ld_Learning_CertifiedGP.ipynb`

Code was run in Julia Version 1.10.2
Please refer to .toml files for package versions. The convergence test was run in a different environment. For .toml files refer to the subfolder `convergence`.
[![DOI](https://zenodo.org/badge/791395881.svg)](https://zenodo.org/doi/10.5281/zenodo.11093644)

![predicted motions with uncertainty quantification](https://github.com/Christian-Offen/Lagrangian_GP/blob/main/continuous/plots/eye_catcher1.png?raw=true "predicted motions with uncertainty quantification")
![predicted Hamiltonian to partially observed system with uncertainty quantification](https://github.com/Christian-Offen/Lagrangian_GP/blob/main/continuous/plots/eye_catcher_2.png?raw=true "predicted Hamiltonian to partially observed system with uncertainty quantification")

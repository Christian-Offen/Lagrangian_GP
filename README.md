# Lagrangian_GP
Accompanying sourcecode for the article

C.Offen
*Machine learning of continuous and discrete variational ODEs with guaranteed convergence and uncertainty quantifications*

To reproduce the experiments of the artcile (and more), run

`continuous/L_Learning_CertifiedGP_RUN.jl`\
`discrete/Ld_Learning_CertifiedGP_RUN.jl`\
`convergence/L_Learning_CertifiedGP_oscillator_1d_convergence_DoubleFloat.jl`

Further experiments to learn discrete Lagrangians may be viewed in\
`discrete/Ld_Learning_CertifiedGP.ipynb`

Code runs in Julia Version 1.9.3.
Please refer to .toml files for package versions in the subfolders.

![predicted motions with uncertainty quantification](https://github.com/Christian-Offen/Lagrangian_GP/blob/main/continuous/plots/eye_catcher1.png?raw=true "predicted motions with uncertainty quantification")
![predicted Hamiltonian to partially observed system with uncertainty quantification](https://github.com/Christian-Offen/Lagrangian_GP/blob/main/continuous/plots/eye_catcher_2.png?raw=true "predicted Hamiltonian to partially observed system with uncertainty quantification")

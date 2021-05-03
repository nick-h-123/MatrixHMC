# ComplexAdvancedHMC.jl

Extendes [AdvancedHMC](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjIgYm7qb_vAhVOGjQIHTblDzAQFjAAegQIAhAD&url=https%3A%2F%2Fgithub.com%2FTuringLang%2FAdvancedHMC.jl&usg=AOvVaw0rhRmTTFWwgGsvKjMhmxfQ) package to complex-valued parameters.

Changes from AdvancedHMC:
- General support for complex-valued parameters while still maintaining most of the support for real-valued parameters.
- Removed support for DenseMatrices.
- Added HermitianMetric to support hermitian parameters.

Dynamic sampling and stopping criterion not supported yet.

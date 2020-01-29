Here we developed the fundamental of the tensor algebra required in a tensor network. We use data generated from the code
and show the concepts inmersed in a tensor network. The organization of the tutorial is:

1. Tensor Contraction
2. Tensor Decomposition
3. Gauge Freedom
4. Canonical Forms
5. Matrix Product States (MPS)
6. Matrix Product Operators (MPO)
7. DMRG algorithm

Additionally, we used two functions implemented by Glen Evenbly described to contract tensors called ncon described in [https://arxiv.org/abs/1402.0939](https://arxiv.org/abs/1402.0939)
and a function to implement DMRG for a 1D chain with open boundaries, using the two-site update strategy, called doDMRG_MPO.

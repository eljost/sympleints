# sympleints
Molecular integrals over shells of Gaussian basis functions using sympy.

Currently, sympleints just generates integral code. It does not provide the
associated machinery to actually evaluate the integrals. For an example how to do
this, please see the module `pysisyphus.wavefunction.shells` in the
[pysisyphus package](https://github.com/eljost/pysisyphus).

The generated python code makes heavy use of `numpy`.

Several types of 1 and 2-electron integrals are implemented.

## Implemented integrals & functions

Evaluation of Gaussian basis functions.

### 1-electron integrals

1. Arbitrary order multipole integrals
  1. Overalp integrals (order 0)
  2. Linear moment (dipole moment) integrals (order 1)
  3. Quadratic moment (quadrupole moment) integrals (order 2)
2. Kinetic energy integrals
3. Nuclear attraction integrals

### 2-electron integrals
1. 2-center-2-electron integrals
2. 3-center-2-electron-integrals

With these two types of 2-electron integrals density fitting can implemented. 
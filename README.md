# sympleints
Molecular integrals over **shells** of Gaussian basis functions using sympy.

By providing the appropriate commands, `sympleints` is able to generate integral code that already includes
the Cartesian-to-spherical transformation (`--sph`) and normalization factors (`--norm-pgto`).

The generated python code makes heavy use of `numpy` and by providing arguments with appropriate
shapes, all functions can be evaluted using arrays of exponents and contraction coefficients
(contraced gaussian functions).

## Installation
```bash
git clone git@github.com:eljost/sympleints.git
cd sympleints
python -m pip install -e .
```

After successful installation the `sympleints` command should be available in the shell.

## Example usage

```bash
$ sympleints --help
usage: sympleints [-h] [--lmax LMAX] [--lauxmax LAUXMAX] [--write] [--out-dir OUT_DIR] [--keys KEYS [KEYS ...]] [--sph] [--opt-basic] [--normalize {pgto,cgto,none}]

options:
  -h, --help            show this help message and exit
  --lmax LMAX           Generate 1e-integrals up to this maximum angular momentum.
  --lauxmax LAUXMAX     Maximum angular moment for integrals using auxiliary functions.
  --write               Write out generated integrals to the current directory, potentially overwriting the present modules.
  --out-dir OUT_DIR     Directory, where the generated integrals are written.
  --keys KEYS [KEYS ...]
                        Generate only certain expressions. Possible keys are: (gto, ovlp, dpm, dqpm, qpm, kin, coul, 2c2e, 3c2e_sph). If not given, all expressions are generated.
  --sph
  --opt-basic           Turn on basic optimizations in CSE.
  --normalize {pgto,cgto,none}
```

To get a quick impression some overlap integrals up to p-functions are generated via:
```bash
sympleints --lmax 1 --keys ovlp
```

The maximum (auxiliary) angular moment is controlled with the `--lmax` and `--lauxmax` arguments.
By default, integrals are generated up to g-functions (`--lmax 4`) and up h-function for the
density fitting integrals (`--lauxmax 5`).

Currently, sympleints just generates integral code. It does not provide the
associated machinery to actually evaluate the integrals. For an example how to do
this, please see the module `pysisyphus.wavefunction.shells` in the
[pysisyphus package](https://github.com/eljost/pysisyphus).

## Implemented integrals & functions

### Functions
  1. Evaluation of shells of Gaussian basis functions

### 1-electron integrals

  1. Arbitrary order multipole integrals
    1. Overalp integrals (order 0)
    2. Linear moment (dipole moment) integrals (order 1)
    3. Quadratic moment (quadrupole moment) integrals (order 2)
    4. Extension to higher multipoles would be trivial
  2. Kinetic energy integrals
  3. Nuclear attraction integrals

### 2-electron integrals
  1. 2-center-2-electron integrals
  2. 3-center-2-electron-integrals

Together, these two types of 2-electron integrals allow the implementation of density fitting.

Evaluation of the Boys-function, as required for the 1- and 2-electron integrals is currently not part
of `sympleints`. Suitable code for the Boys-function is found in the
`pysisyphus.wavefunction.ints.boys` module. The code will be made a part of this package at a later date.

## Advantages
The resulting Python code is surely not optimal, but very convenient to have and easy to use. Besides an implementation
of the Boys-function, the resulting code only depends numpy. Understanding the integral generator is still possible.

Calculation of the overlap matrix with 1300 spherical basis functions (Tris(bipyridine)ruthenium(II))
takes 9.5 s, calculation of the quadratic moment integrals requires 20.2 s. Maybe, these timings make the previous sentence more suitable for its placement in the Limiations section ;)

## Limitations
Generation of the 3-center-2-electron integrals up to the g and h-function is slow (2 hours)! Generation of the actual sympy expressions is very fast, but subsequent simplification, common subexpression elimination and variable substitution takes ages with sympy.

I did not yet test, if Python 3.11 yields any performance benefits.

Right now, code for all possible combinations of angular moment is generated even though one combination for each pair/triple of angular momenta would be enough. Having an (sp)-overlap integral should be enough to also evaluate an (ps)-overlap integral.

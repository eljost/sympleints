{ buildPythonPackage
, lib
, gfortran
, setuptools
, setuptools-scm
, meson-python
, black
, colorama
, fprettify
, jinja2
, matplotlib
, networkx
, numpy
, pathos
, psutil
, sympy
, scipy
, pyscf
, pytestCheckHook
}:

buildPythonPackage rec {
  pname = "sympleints";
  version = "0.1.0";

  src = lib.cleanSource ../.;

  pyproject = true;

  nativeBuildInputs = [ gfortran ];

  build-system = [
    meson-python
  ];

  dependencies = [
    black
    colorama
    fprettify
    jinja2
    matplotlib
    networkx
    numpy
    pathos
    psutil
    sympy
    scipy
  ];

  nativeCheckInputs = [
    pytestCheckHook
  ];

  checkInputs = [    
    pyscf
  ];

  meta = with lib; {
    description = "Molecular integrals over Gaussian basis functions using sympy";
    license = licenses.eupl12;
    homepage = "https://github.com/eljost/sympleints";
    maintainers = [ maintainers.sheepforce ];
  };
}

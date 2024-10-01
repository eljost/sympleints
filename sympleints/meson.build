sources = files(
  '__init__.py',
  'benchmark.py',
  'cart2sph.py',
  'config.py',
  'FortranRenderer.py',
  'Functions.py',
  'helpers.py',
  'l_iters.py',
  'logger.py',
  'main.py',
  'NumbaRenderer.py',
  'patch_sympy.py',
  'PythonRenderer.py',
  'Renderer.py',
  'symbols.py',
  'sym_solid_harmonics.py',
)

python.install_sources(
  sources,
  subdir: 'sympleints'
)

subdir('defs')
subdir('templates')
subdir('testing')
subdir('graphs')

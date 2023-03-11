project('sympleints', 'c',
  version : '0.0.1',
  #default_options : ['warning_level=2']
)

add_languages('fortran')

subdir('sympleints/fortran')

#py_mod = import('python')
#py3 = py_mod.find_installation('python3')
#py3_dep = py3.dependency()
#message(py3.path())
#message(py3.get_install_dir())

#incdir_numpy = run_command(py3,
  #['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  #check : true
#).stdout().strip()
#message('numpy include:', incdir_numpy)


#incdir_f2py = run_command(py3,
    #['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    #check : true
#).stdout().strip()
#message('f2py include:', incdir_numpy)

#f_source_dir = meson.source_root() + '/sympleints/fortran/'
#message('f_source_dir', f_source_dir)

#f_source = custom_target('fintor.c',
                            #input : ['fintor.f90'],
                            #output : ['fintormodule.c', 'fintor-f2pywrappers2.f90'],
                            #command : [ py3, '-m', 'numpy.f2py', '@INPUT@',
                            #'-m', 'fintor', '--lower'])

#inc_np = include_directories(incdir_numpy, incdir_f2py)

#py3.extension_module('fibby',
           #'fib1.f',
           #fibby_source,
           #incdir_f2py+'/fortranobject.c',
           #include_directories: inc_np,
           #dependencies : py3_dep,
           #install : true)

import functools
import itertools as it
import random
import string
import textwrap
import time

from jinja2 import Template
from sympy.codegen.ast import Assignment
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.c import C99CodePrinter
from sympy.printing.fortran import FCodePrinter

from sympleints.helpers import shell_shape, shell_shape_iter


def func_name_from_Ls(name, Ls):
    return name + "_" + "".join(str(l) for l in Ls)


def make_py_func(
    repls, reduced, shape, shape_iter, args=None, name=None, doc_str="", multidim=True
):
    if args is None:
        args = list()
    # Generate random name, if no name was supplied
    if name is None:
        name = "func_" + "".join(
            [random.choice(string.ascii_letters) for i in range(8)]
        )

    # This allows using the 'boys' function without producing an error
    print_settings = {
        "allow_unknown_functions": True,
    }
    print_func = NumPyPrinter(print_settings).doprint
    assignments = [Assignment(lhs, rhs) for lhs, rhs in repls]
    py_lines = [print_func(as_) for as_ in assignments]
    result_lines = [print_func(red) for red in reduced]
    # With 'multidim = True' we expect the orbital exponents and the contraction
    # coefficients to be 2d/3d/... numpy array. Then we can utilize array broadcasting
    # to evalute the integrals over products of primitive basis functions.
    if multidim:
        result_lines = [f"numpy.sum({line})" for line in result_lines]

    # shape_iter must be modified, if multiple components are present.
    # This calculation should probably be done outside the function, so we don't
    # have to reimplement this for the other languages.
    per_component = functools.reduce(lambda a, b: a * b, shape, 1)
    components = len(reduced) // per_component
    if components > 1:
        shape = (components, *shape)
        shapes = list(shape_iter)
        new_shape_iter = list()
        for comp in range(components):
            for shape_ in shapes:
                new_shape_iter.append((comp, *shape_))
        shape_iter = new_shape_iter

    results_iter = zip(shape_iter, result_lines)

    tpl = Template(
        """
    def {{ name }}({{ args }}):
        {% if doc_str %}
        \"\"\"{{ doc_str }}\"\"\"
        {% endif %}

        result = numpy.zeros({{ shape }}, dtype=float)

        {% for line in py_lines %}
        {{ line }}
        {% endfor %}

        # {{ n_return_vals }} item(s)
        {% for inds, res_line in results_iter %}
        result[{{ inds|join(", ")}}] = {{ res_line }}
        {% endfor %}
        return result
    """,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    rendered = textwrap.dedent(
        tpl.render(
            name=name,
            args=args,
            py_lines=py_lines,
            results_iter=results_iter,
            n_return_vals=len(reduced),
            doc_str=doc_str,
            shape=shape,
        )
    ).strip()
    return rendered


def make_py_dispatch_func(name, args, py_func_map, L_num):
    """TODO: Should probably be switched to match/case or dict lookup."""
    args_str = ", ".join([str(arg) for arg in args])
    # Ls_args_str = "Ls, " + args_str
    tpl = Template(
        """
    def {{ name }}(Ls, {{ args_str }}):
        {% for Ls, func_name in func_map %}
            {% if loop.index0 == 0 %}
            if Ls == {{ Ls }}:
            {% else %}
            elif Ls == {{ Ls }}:
            {% endif %}
                return {{ func_name }}({{ args_str }})
        {% endfor %}
            # Guard for numba, so it picks up the right return type of the function.
            # Without this guard, numba only recognizes an OptionalType.
            # When the body of the else-clause actually runs, AssertionError
            # will be raised.
            else:
                for L in Ls:
                    assert 0 <= L <= _L_MAX
                return numpy.zeros((0, 0))
    """,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    rendered = textwrap.dedent(
        tpl.render(name=name, args_str=args_str, func_map=py_func_map)
    ).strip()
    return rendered


def make_func_dict(name, func_map):
    tpl = Template(
        """
        {{ name }} = {
        {% for Ls, func_name in func_map %}
        {{ Ls }}: {{ func_name }},
        {% endfor %}
        }
        """,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    rendered = textwrap.dedent(tpl.render(name=name, func_map=func_map)).strip()
    return rendered


def render_py_module(funcs, add_imports, L_max, comment):
    tpl = Template(
        "{{ comment }}\n\nimport numpy\n\n{{ add_imports }}\n\n"
        "_L_MAX = {{ L_max }}\n\n{{ funcs }}",
        trim_blocks=True,
        lstrip_blocks=True,
    )

    joined = "\n\n".join(funcs)
    add_imports_str = "\n".join(add_imports)
    if add_imports:
        add_imports_str += "\n\n"
    rendered = tpl.render(
        comment=comment,
        add_imports=add_imports_str,
        L_max=L_max,
        funcs=joined,
    )
    rendered = textwrap.dedent(rendered)
    try:
        import black

        try:
            rendered = black.format_str(rendered, mode=black.FileMode(line_length=90))
        except black.parsing.InvalidInput:
            print("Error while parsing with black. Dumping nontheless.")
    except ModuleNotFoundError:
        print("Skipped formatting with black, as it is not installed!")
    return rendered


def render_py_funcs(exprs_Ls, args, base_name, doc_func, add_imports=None):
    args = ", ".join((map(str, args)))
    if add_imports is None:
        add_imports = ()

    funcs = list()
    func_map = list()
    for (repls, reduced), L_tots in exprs_Ls:
        shape = shell_shape(L_tots, cartesian=True)
        shape_iter = shell_shape_iter(L_tots, cartesian=True)
        doc_str = doc_func(L_tots)
        doc_str += "\n\nGenerated code; DO NOT modify by hand!"
        name = func_name_from_Ls(base_name, L_tots)
        func_map.append((L_tots, name))
        print(f"Rendering '{name}' ... ", end="")
        start = time.time()
        func = make_py_func(
            repls, reduced, shape, shape_iter, args=args, name=name, doc_str=doc_str
        )
        dur = time.time() - start
        print(f"finished in {dur: >8.2f} s")
        funcs.append(func)
    return funcs, func_map


def make_c_func(repls, reduced, args=None, name=None, doc_str=""):
    if args is None:
        args = list()
    # Generate random name, if no name was supplied
    if name is None:
        name = "func_" + "".join(
            [random.choice(string.ascii_letters) for i in range(8)]
        )
    arg_strs = list()
    for arg in args:
        if arg.islower():
            arg_str = f"double {arg}"
        elif arg.isupper():
            arg_str = f"double {arg}[3]"
        else:
            raise Exception
        arg_strs.append(arg_str)
    args_str = ", ".join(arg_strs)

    # This allows using the 'boys' function without producing an error
    print_settings = {
        "allow_unknown_functions": True,
        # Without disabling contract some expressions will raise ValueError.
        "contract": False,
    }
    print_func = C99CodePrinter(print_settings).doprint
    assignments = [Assignment(lhs, rhs) for lhs, rhs in repls]
    repl_lines = [print_func(as_) for as_ in assignments]
    res_lines = [print_func(red) for red in reduced]
    res_len = len(reduced)

    signature = f"double * {name}({args_str})"

    tpl = Template(
        """
    {{ signature }} {
        {% if doc_str %}
        /* {{ doc_str }} */
        {% endif %}

        static double {{ res_name }}[{{ res_len}}];

        {% for line in repl_lines %}
        const double {{ line }}
        {% endfor %}

        {% for rhs in res_lines %}
        {{ res_name }}[{{ loop.index0}}] = {{ rhs }};
        {% endfor %}

        return {{ res_name }};
    }
    """,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    rendered = textwrap.dedent(
        tpl.render(
            signature=signature,
            res_name="results",
            res_len=res_len,
            repl_lines=repl_lines,
            res_lines=res_lines,  # c_lines=c_lines,
            reduced=reduced,
            doc_str=doc_str,
            args_str=args_str,
        )
    ).strip()
    return rendered, signature


def render_c_funcs(exprs_Ls, args, base_name, doc_func, add_imports=None, comment=""):
    if add_imports is not None:
        raise Exception("Implement me!")

    arg_strs = [str(arg) for arg in args]

    funcs = list()
    signatures = list()
    for (repls, reduced), L_tots in exprs_Ls:
        doc_str = doc_func(L_tots)
        doc_str += "\n\n\t\tGenerated code; DO NOT modify by hand!"
        doc_str = textwrap.dedent(doc_str)
        name = func_name_from_Ls(base_name, L_tots)
        func, signature = make_c_func(
            repls, reduced, args=arg_strs, name=name, doc_str=doc_str
        )
        funcs.append(func)
        signatures.append(signature)
    funcs_joined = "\n\n".join(funcs)

    if comment != "":
        comment = f"/*{comment}*/"

    # Render C files
    c_tpl = Template(
        "#include <math.h>\n\n{{ comment }}\n\n{{ funcs }}",
        trim_blocks=True,
        lstrip_blocks=True,
    )
    c_rendered = c_tpl.render(comment=comment, funcs=funcs_joined)
    c_rendered = textwrap.dedent(c_rendered)
    # Render simple header file
    h_rendered = "\n".join([f"{sig};" for sig in signatures])
    return c_rendered, h_rendered


class FCodePrinterMod(FCodePrinter):
    def _print_Indexed(self, expr):
        # prints I[0] as I[1], i.e., increments the index by one.
        inds = [self._print(i + 1) for i in expr.indices]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))


def make_fortran_comment(comment_str):
    return "".join([f"! {line}\n" for line in comment_str.strip().split("\n")])


def make_f_func(repls, reduced, shape, shape_iter, args=None, name=None, doc_str=""):
    if args is None:
        args = list()
    # Generate random name, if no name was supplied
    if name is None:
        name = "func_" + "".join(
            [random.choice(string.ascii_letters) for i in range(8)]
        )
    args_str = ", ".join(args)
    exp_args = [arg for arg in args if arg.islower()]
    center_args = [arg for arg in args if arg.isupper()]
    # This allows using the 'boys' function without producing an error
    print_settings = {
        "allow_unknown_functions": True,
        # Without disabling contract some expressions will raise ValueError.
        "contract": False,
        "standard": 2008,
        "source_format": "free",
    }
    print_func = FCodePrinterMod(print_settings).doprint
    assignments = [Assignment(lhs, rhs) for lhs, rhs in repls]
    repl_lines = [print_func(as_) for as_ in assignments]
    results = [print_func(red) for red in reduced]
    shape_iter_plus = [[i + 1 for i in inds] for inds in shape_iter]
    results_iter = zip(shape_iter_plus, results)
    res_len = len(reduced)
    res_name = "res"
    res_dims = ", ".join([":"] * len(shape))

    doc_str = make_fortran_comment(doc_str)

    tpl = Template(
        """subroutine {{ name }} ({{ args_str }}, {{ res_name }})

        {% if doc_str %}
        {{ doc_str }}
        {% endif %}

        ! Arguments provided by the user
        {% for arg in exp_args %}
        real({{ kind }}), intent(in) :: {{ arg }}  ! Primitive exponent
        {% endfor %}
        {% for arg in center_args %}
        real({{ kind  }}), intent(in), dimension(3) :: {{ arg }}  ! Center
        {% endfor %}
        ! Return value
        real({{ kind }}), intent(in out) :: {{ res_name }}({{ res_dims }})

        ! Intermediate quantities
        {% for as_ in assignments %}
        real({{ kind }}) :: {{ as_.lhs }}
        {% endfor %}

        {% for line in repl_lines %}
        {{ line }}
        {% endfor %}

        {% for inds, res_line in results_iter %}
        {{ res_name }}({{ inds|join(", ") }})  = {{ res_line }}
        {% endfor %}
    end subroutine {{ name }}
    """,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    rendered = textwrap.dedent(
        tpl.render(
            name=name,
            exp_args=exp_args,
            center_args=center_args,
            res_name=res_name,
            res_dims=res_dims,
            res_len=res_len,
            assignments=assignments,
            repl_lines=repl_lines,
            shape=shape,
            results_iter=results_iter,
            reduced=reduced,
            doc_str=doc_str,
            args_str=args_str,
            # Using 'kind=real64' does not seem to work with f2py, even with
            # .f2py_f2cmap. Using only '8' seems to work though.
            kind="kind=real64",
        )
    ).strip()
    return rendered


def make_f_dispatch_func(name, args, func_map, L_num, sph):
    Ls = [f"L{ind}" for _, ind in zip(range(L_num), ("a", "b", "c", "d"))]
    exponents = ("ax", "bx", "cx", "dx")[:L_num]
    centers = ("A", "B", "C", "D")
    Ls_str = ", ".join(Ls)
    args_str = ", ".join([str(arg) for arg in args])
    res_name = "res"

    if sph:
        dims = [f"(2*{L} + 1" for L in Ls]
    else:
        dims = [f"({L}+2)*({L}+1)/2" for L in Ls]
    res_dim = ", ".join(dims)

    comps_funcs = list()
    for ang_moms, func_name in func_map:
        condition = (
            "("
            + " .and. ".join([f"({L} == {angmom})" for L, angmom in zip(Ls, ang_moms)])
            + ")"
        )
        comps_funcs.append((condition, func_name))

    tpl = Template(
        """
    function {{ name }}({{ Ls_str }}, {{ args_str }}) result({{ res_name }})
        integer, intent(in) :: {{ Ls_str }}
        real(kind=real64), intent(in) :: {{ exponents|join(", ") }}
        real(kind=real64), intent(in), dimension(3) :: {{ centers|join(", ") }}
        real(kind=real64), dimension({{ res_dim }}) :: res

        {% for comp, func_name in comps_funcs %}
            {% if loop.index0 == 0 %}
            if {{ comp }} then
            {% elif loop.last %}
            else
            {% else %}
            else if {{ comp }} then
            {% endif %}
                call {{ func_name }}({{ args_str }}, res)
        {% endfor %}
            end if
    end function {{ name }}
    """,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    rendered = textwrap.dedent(
        tpl.render(
            name=name,
            res_name=res_name,
            comps_funcs=comps_funcs,
            Ls_str=Ls_str,
            exponents=exponents,
            centers=centers,
            res_dim=res_dim,
            args_str=args_str,
        )
    ).strip()
    return rendered


def render_f_funcs(exprs_Ls, args, base_name, doc_func, add_imports=None, comment=""):
    if add_imports is not None:
        raise Exception("Implement me!")

    arg_strs = [str(arg) for arg in args]

    funcs = list()
    for (repls, reduced), L_tots in exprs_Ls:
        shape = shell_shape(L_tots, cartesian=True)
        shape_iter = shell_shape_iter(L_tots, cartesian=True)
        doc_str = doc_func(L_tots)
        doc_str += "\n\n\t\tGenerated code; DO NOT modify by hand!"
        doc_str = textwrap.dedent(doc_str)
        name = func_name_from_Ls(base_name, L_tots)
        func = make_f_func(
            repls, reduced, shape, shape_iter, args=arg_strs, name=name, doc_str=doc_str
        )
        funcs.append(func)
    return funcs


def render_f_module(funcs, base_name, comment=""):
    if comment != "":
        comment = make_fortran_comment(comment)

    funcs_joined = "\n\n".join(funcs)

    # Render F files
    mod_name = f"mod_{base_name}"
    f_tpl = Template(
        # Using iso_fortran_env is not needed w/ simple kind=8
        # "module {{ base_name }}\n\nuse iso_fortran_env, only: real64\n\nimplicit none\n\n"
        "module {{ mod_name }}\n\nuse iso_fortran_env, only: real64\n\n"
        "implicit none\n\ncontains\n\n{{ funcs }}\n\nend module {{ mod_name }}",
        trim_blocks=True,
        lstrip_blocks=True,
    )
    f_rendered = f_tpl.render(mod_name=mod_name, comment=comment, funcs=funcs_joined)
    f_rendered = textwrap.dedent(f_rendered)
    # Render simple header file
    return f_rendered


def write_file(out_dir, name, rendered):
    out_name = out_dir / name
    with open(out_name, "w") as handle:
        handle.write(rendered)
    print(f"Wrote '{out_name}'.")


def write_render(
    ints_Ls,
    args,
    name,
    doc_func,
    out_dir,
    comment="",
    py_kwargs=None,
    c=True,
    f=True,
    c_kwargs=None,
    f_kwargs=None,
):
    if comment != "":
        comment = f'"""\n{comment}\n"""\n\n'
    if py_kwargs is None:
        py_kwargs = {}
    if c_kwargs is None:
        c_kwargs = {}
    if f_kwargs is None:
        f_kwargs = {}
    ints_Ls = list(ints_Ls)
    Ls = [ls for _, ls in ints_Ls]
    L_max = max(*it.chain(*Ls))
    L_num = len(Ls[0])

    # Python
    py_imports = py_kwargs.get("add_imports", tuple())
    py_funcs, func_map = render_py_funcs(ints_Ls, args, name, doc_func, **py_kwargs)
    """
    py_dispatch = make_py_dispatch_func(
        name,
        args,
        func_map,
        L_num,
    )
    py_funcs = [
        py_dispatch,
    ] + py_funcs
    """
    py_func_dict = make_func_dict(name, func_map)
    py_funcs = py_funcs + [
        py_func_dict,
    ]
    # Pure python/numpy
    np_rendered = render_py_module(py_funcs, py_imports, L_max, comment)
    write_file(out_dir, f"{name}.py", np_rendered)

    # Numba, jitted
    jit_funcs = [f"@jit(nopython=True, nogil=True)\n{func}" for func in py_funcs]
    numba_rendered = render_py_module(
        jit_funcs, py_imports + ("from numba import jit",), L_max, comment
    )
    write_file(out_dir, f"{name}_numba.py", numba_rendered)

    # C
    if c:
        c_rendered, h_rendered = render_c_funcs(
            ints_Ls,
            args,
            name,
            doc_func,
            comment=comment,
            **c_kwargs,
        )
        write_file(out_dir, f"{name}.c", c_rendered)
        write_file(out_dir, f"{name}.h", h_rendered)

    # Fortran
    if f:
        f_funcs = render_f_funcs(
            ints_Ls,
            args,
            name,
            doc_func,
            comment=comment,
            **f_kwargs,
        )
        f_dispatch = make_f_dispatch_func(
            name,
            args,
            func_map,
            L_num,
            sph=False,
        )
        f_funcs = [
            f_dispatch,
        ] + f_funcs
        f_rendered = render_f_module(f_funcs, name, comment="")
        write_file(out_dir, f"{name}.f90", f_rendered)

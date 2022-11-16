import random
import string
import textwrap
import time

from jinja2 import Template
from sympy.codegen.ast import Assignment
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.c import C99CodePrinter
from sympy.printing.fortran import FCodePrinter


class FCodePrinterMod(FCodePrinter):

    def _print_Indexed(self, expr):
        # prints I[0] as I[1], i.e., increments the index by one.
        inds = [ self._print(i+1) for i in expr.indices ]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))


def make_py_func(repls, reduced, args=None, name=None, doc_str=""):
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
    return_val = print_func(reduced)

    tpl = Template(
        """
    def {{ name }}({{ args }}):
        {% if doc_str %}
        \"\"\"{{ doc_str }}\"\"\"
        {% endif %}

        {% for line in py_lines %}
        {{ line }}
        {% endfor %}

        # {{ n_return_vals }} item(s)
        return numpy.array({{ return_val }})
    """,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    rendered = textwrap.dedent(
        tpl.render(
            name=name,
            args=args,
            py_lines=py_lines,
            return_val=return_val,
            n_return_vals=len(reduced),
            doc_str=doc_str,
        )
    ).strip()
    return rendered


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


def make_f_func(repls, reduced, args=None, name=None, doc_str=""):
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
    res_lines = [print_func(red) for red in reduced]
    res_len = len(reduced)
    res_name = "res"

    doc_str = "".join([f"! {line}\n" for line in doc_str.strip().split("\n")])

    # TODO: name mangling leads to A -> A_ and B -> B_ etc. conversions, as 
    # Fortran is case insensitive.
    tpl = Template("""pure function {{ name }} ({{ args_str }}) result ( {{res_name}} )

        {% if doc_str %}
        {{ doc_str }}
        {% endif %}

        ! Arguments provided by the user
        {% for arg in exp_args %}
        real, intent(in) :: {{ arg }}  ! Primitive exponent
        {% endfor %}
        {% for arg in center_args %}
        real, intent(in), dimension(3) :: {{ arg }}  ! Center
        {% endfor %}
        ! Intermediate quantities
        {% for as_ in assignments %}
        real :: {{ as_.lhs }}
        {% endfor %}
        ! Return value
        real, dimension({{ res_len }}) :: {{ res_name }}

        {% for line in repl_lines %}
        {{ line }}
        {% endfor %}

        {% for rhs in res_lines %}
        {{ res_name }}({{ loop.index0+1}}) = {{ rhs }}
        {% endfor %}
    end function {{ name }}
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
            res_len=res_len,
            assignments=assignments,
            repl_lines=repl_lines,
            res_lines=res_lines,  # c_lines=c_lines,
            reduced=reduced,
            doc_str=doc_str,
            args_str=args_str,
        )
    ).strip()
    return rendered


def render_py_funcs(exprs_Ls, args, base_name, doc_func, add_imports=None, comment=""):
    if add_imports is None:
        add_imports = ()
    add_imports_str = "\n".join(add_imports)
    if add_imports:
        add_imports_str += "\n\n"

    args = ", ".join((map(str, args)))

    py_funcs = list()
    for (repls, reduced), L_tots in exprs_Ls:
        doc_str = doc_func(L_tots)
        doc_str += "\n\nGenerated code; DO NOT modify by hand!"
        name = base_name + "_" + "".join(str(l) for l in L_tots)
        print(f"Rendering '{name}' ... ", end="")
        start = time.time()
        py_func = make_py_func(repls, reduced, args=args, name=name, doc_str=doc_str)
        dur = time.time() - start
        print(f"finished in {dur: >8.2f} s")
        py_funcs.append(py_func)
    py_funcs_joined = "\n\n".join(py_funcs)

    if comment != "":
        comment = f'"""\n{comment}\n"""\n\n'

    np_tpl = Template(
        "import numpy\n\n{{ add_imports }}{{ comment }}{{ py_funcs }}",
        trim_blocks=True,
        lstrip_blocks=True,
    )
    np_rendered = np_tpl.render(
        comment=comment, add_imports=add_imports_str, py_funcs=py_funcs_joined
    )
    np_rendered = textwrap.dedent(np_rendered)
    try:
        from black import format_str, FileMode

        np_rendered = format_str(np_rendered, mode=FileMode(line_length=90))
    except ModuleNotFoundError:
        print("Skipped formatting with black, as it is not installed!")

    return np_rendered


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
        name = base_name + "_" + "".join(str(l) for l in L_tots)
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


def render_f_funcs(exprs_Ls, args, base_name, doc_func, add_imports=None, comment=""):
    if add_imports is not None:
        raise Exception("Implement me!")

    arg_strs = [str(arg) for arg in args]

    funcs = list()
    for (repls, reduced), L_tots in exprs_Ls:
        doc_str = doc_func(L_tots)
        doc_str += "\n\n\t\tGenerated code; DO NOT modify by hand!"
        doc_str = textwrap.dedent(doc_str)
        name = base_name + "_" + "".join(str(l) for l in L_tots)
        func = make_f_func(
            repls, reduced, args=arg_strs, name=name, doc_str=doc_str
        )
        funcs.append(func)
    funcs_joined = "\n\n".join(funcs)

    if comment != "":
        raise Exception("Comments for Fortran are not yet implemented!")

    # Render F files
    f_tpl = Template(
        "module {{ base_name }}\n\nimplicit none\n\n"
        "contains\n\n{{ funcs }}\n\nend module {{ base_name }}",
        trim_blocks=True,
        lstrip_blocks=True,
    )
    f_rendered = f_tpl.render(base_name=base_name, comment=comment, funcs=funcs_joined)
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
    if py_kwargs is None:
        py_kwargs = {}
    if c_kwargs is None:
        c_kwargs = {}
    if f_kwargs is None:
        f_kwargs = {}
    ints_Ls = list(ints_Ls)
    # Python
    py_rendered = render_py_funcs(
        ints_Ls, args, name, doc_func, comment=comment, **py_kwargs
    )
    write_file(out_dir, f"{name}.py", py_rendered)
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
    if f:
        f_rendered = render_f_funcs(
            ints_Ls,
            args,
            name,
            doc_func,
            comment=comment,
            **f_kwargs,
        )
        write_file(out_dir, f"{name}.f90", f_rendered)

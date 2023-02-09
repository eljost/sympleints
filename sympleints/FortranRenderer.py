import os
import re
import tempfile
import textwrap

from jinja2 import Template
from sympy.codegen.ast import Assignment
from sympy.printing.fortran import FCodePrinter

from sympleints.Renderer import Renderer
from sympleints.helpers import shell_shape_iter


class FCodePrinterMod(FCodePrinter):
    boys_re = re.compile("boys\(([d\d\.]+),(.+)")

    def _print_Function(self, expr):
        func = super()._print_Function(expr)
        if func.startswith("boys"):
            mobj = self.boys_re.match(func)
            as_float = float(mobj.group(1).lower().replace("d", "e"))
            as_int = int(as_float)
            remainder = mobj.group(2)
            func = f"boys({as_int},{remainder}"
        return func

    def _print_Indexed(self, expr):
        # prints I[0] as I[1], i.e., increments the index by one.
        inds = [self._print(i + 1) for i in expr.indices]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))


def make_fortran_comment(comment_str):
    return "".join([f"! {line}\n" for line in comment_str.strip().split("\n")])


class FortranRenderer(Renderer):

    ext = ".f90"
    real_kind = "kind=real64"
    res_name = "res"

    def shell_shape_iter(self, *args, **kwargs):
        # Start indexing at 1, instead of 0.
        return shell_shape_iter(*args, start_at=1, **kwargs)

    def get_argument_declaration(self, functions, contracted=False):
        tpl = Template(
            textwrap.dedent(
                """
            {% for exp_ in exps %}
            real({{ kind }}), intent(in) :: {{ exp_ }}{% if contracted %}(:){% endif %}  ! Primitive exponent(s)
            {% endfor %}
            {% if contracted %}
            ! Contraction coefficient(s)
            {% for exp_, coeff in zip(exps, coeffs) %}
            real({{ kind }}), intent(in), dimension(size({{ exp_ }})) :: {{ coeff }}
            {% endfor %}
            {% else %}
            real({{ kind }}), intent(in) :: {{ coeffs|join(", ") }}
            {% endif %}
            ! Center(s)
            real({{ kind }}), intent(in), dimension(3) :: {{ centers|join(", ") }}
            {% if ref_center %}
            ! Reference center; used only by some procedures
            real({{ kind }}), intent(in), dimension(3) :: {{ ref_center }}
            {% endif %}
            ! Return value
            real({{ kind }}), intent(in out) :: {{ res_name }}({{ res_dim }})
            """
            ),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        arg_dim = "(:)" if contracted else ""
        res_dim = ", ".join(":" for _ in range(functions.ndim))
        rendered = tpl.render(
            kind=self.real_kind,
            exps=functions.exponents,
            contracted=contracted,
            arg_dim=arg_dim,
            coeffs=functions.coeffs,
            zip=zip,
            centers=functions.centers,
            ref_center=functions.ref_center
            if (contracted or functions.with_ref_center)
            else None,
            res_name=self.res_name,
            res_dim=res_dim,
        )
        return rendered

    def render_function(
        self, functions, repls, reduced, shape, shape_iter, args, name, doc_str=""
    ):
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
        res_len = len(reduced)
        results_iter = zip(shape_iter, results)

        doc_str = make_fortran_comment(doc_str)
        arg_declaration = self.get_argument_declaration(functions)

        tpl = Template(
            textwrap.dedent(
                """
            subroutine {{ name }} ({{ args|join(", ") }}, {{ res_name }})

            {% if doc_str %}
            {{ doc_str }}
            {% endif %}

            {{ arg_declaration }}

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
        """
            ),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        rendered = textwrap.dedent(
            tpl.render(
                name=name,
                args=functions.full_args,
                doc_str=doc_str,
                arg_declaration=arg_declaration,
                res_name=self.res_name,
                res_len=res_len,
                assignments=assignments,
                repl_lines=repl_lines,
                shape=shape,
                results_iter=results_iter,
                reduced=reduced,
                kind=self.real_kind,
            )
        ).strip()
        return rendered

    def render_f_init(self, name, rendered_funcs, func_array_name="func_array"):
        F_INIT_TPL = Template(
            """
        subroutine {{ name }}_init ()
        ! Initializer procedure. MUST be called before {{ name }} can be called.
        {% for func in funcs %}
            {{ func_array_name }}({{ func.Ls|join(", ") }})%f => {{ func.name }}
        {% endfor %}
        end subroutine
        """,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        f_init = F_INIT_TPL.render(
            name=name,
            funcs=rendered_funcs,
            func_array_name=func_array_name,
        )
        return f_init

    def render_contracted_driver(self, functions):
        tpl = Template(
            textwrap.dedent(
                """
        subroutine {{ name }}({{ L_args|join(", ") }}, {{ args|join(", ") }})
            integer, intent(in) :: {{ Ls|join(", ") }}
            {{ arg_declaration }}
            real(kind=real64), allocatable, dimension({{ res_dim }}) :: res_tmp
            ! Initializing with => null () adds an implicit save, which will mess
            ! everything up when running with OpenMP.
            procedure({{ name }}_proc), pointer :: fncpntr
            integer :: {{ loop_counter|join(", ") }}

            allocate(res_tmp, mold=res)
            fncpntr => func_array({{ L_args|join(", ") }})%f

            res = 0
            {{ loops|join("\n") }}
                call fncpntr({{ pntr_args }}, res_tmp)
                res = res + res_tmp
            {% for _ in loops %}
            end do
            {% endfor %}
            deallocate(res_tmp)
        end subroutine {{ name }}
        """
            ).strip(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        args = functions.prim_args + [functions.ref_center, self.res_name]
        arg_declaration = self.get_argument_declaration(functions, contracted=True)
        res_dim = ", ".join(":" for _ in range(functions.ndim))

        nbfs = functions.nbfs
        counters = ["i", "j", "k", "l"][:nbfs]

        def loop_iter(counters, exps, coeffs):
            loops = list()
            for counter, exp_ in zip(counters, exps):
                loop = f"do {counter} = 1, size({exp_})"
                loops.append(loop)
            return loops

        loops = loop_iter(counters, functions.exponents, functions.coeffs)
        pntr_args = list()
        for counter, exp_, coeff, center in zip(
            counters, functions.exponents, functions.coeffs, functions.centers
        ):
            pntr_args.append(f"{exp_}({counter}), {coeff}({counter}), {center}")
        pntr_args = ", ".join(pntr_args)
        if functions.with_ref_center:
            pntr_args += f", {functions.ref_center}"

        rendered = tpl.render(
            name=functions.name,
            arg_declaration=arg_declaration,
            res_dim=res_dim,
            L_args=functions.L_args,
            Ls=functions.Ls,
            loop_counter=counters,
            loops=loops,
            pntr_args=pntr_args,
            args=args,
        )
        return rendered

    def render_module(self, functions, rendered_funcs):
        comment = make_fortran_comment(functions.comment)
        header = make_fortran_comment(functions.header)

        init = self.render_f_init(functions.name, rendered_funcs)
        name = functions.name
        mod_name = f"mod_{name}"
        interface_name = f"{name}_proc"
        arg_declaration = self.get_argument_declaration(functions)
        # TODO: handle l_aux_max ...
        l_max = functions.l_max
        func_arr_dims = [f"0:{l_max}" for _ in range(functions.nbfs)]
        contr_driver = self.render_contracted_driver(functions)

        f_tpl = Template(
            textwrap.dedent(
                """
            {{ header }}
            module {{ mod_name }}

            use iso_fortran_env, only: real64
            {% if boys %}
            use mod_boys, only: boys
            {% endif %}

            implicit none

            type fp
                procedure({{ interface_name }}) ,pointer ,nopass :: f =>null()
            end type fp

            interface
                subroutine {{ interface_name }}({{ args|join(", ")}})
                  import :: real64
                  {{ arg_declaration }}
                end subroutine {{ interface_name }}
            end interface

            type(fp) :: func_array({{ func_arr_dims|join(", ") }})

            contains
            {{ init }}

            {{ contr_driver }}

            {% for func in funcs %}
                {{ func.text }}
            {% endfor %}


            end module {{ mod_name }}
            """.strip()
            ),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        rendered = f_tpl.render(
            header=header,
            mod_name=mod_name,
            boys=functions.boys,
            args=functions.full_args + [self.res_name],
            interface_name=interface_name,
            contr_driver=contr_driver,
            arg_declaration=arg_declaration,
            comment=comment,
            func_arr_dims=func_arr_dims,
            init=init,
            funcs=rendered_funcs,
        )
        rendered = textwrap.dedent(rendered)
        rendered_backup = rendered
        # I wonder if fprettify can also deal with strings instead of files only?!
        # I did not manage to find out ... so we use some temporary files here.
        try:
            import fprettify

            try:
                fp = tempfile.NamedTemporaryFile("w", delete=False)
                fp.write(rendered)
                fp.close()
                fprettify.reformat_inplace(
                    fp.name
                )  # (infile=infile, outfile=outfile, orig_filename="rendered")
                with open(fp.name) as handle:
                    rendered = handle.read()
                os.remove(fp.name)
                print("\t ... formatted Fortran code with fprettify")
            except fprettify.FprettifyException as err:
                print("Error while running fprettify. Dumping nontheless.")
                rendered = rendered_backup
        except ModuleNotFoundError:
            print("Skipped formatting with fprettify, as it is not installed!")
        return rendered

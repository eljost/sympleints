import functools
import itertools as it
import random
import string
import textwrap
import time

from jinja2 import Template
from sympy.codegen.ast import Assignment
from sympy.printing.numpy import NumPyPrinter

from sympleints.Renderer import Renderer


class PythonRenderer(Renderer):

    ext = ".py"
    language = "Python"

    def render_function(
        self,
        functions,
        repls,
        reduced,
        shape,
        shape_iter,
        args,
        name,
        doc_str="",
    ):
        # This allows using the 'boys' function without producing an error
        print_settings = {
            "allow_unknown_functions": True,
        }
        print_func = NumPyPrinter(print_settings).doprint

        args = ", ".join([str(arg) for arg in args])
        assignments = [Assignment(lhs, rhs) for lhs, rhs in repls]
        py_lines = [print_func(as_) for as_ in assignments]
        result_lines = [print_func(red) for red in reduced]
        # Here, we expect the orbital exponents and the contraction
        # coefficients to be 2d/3d/... numpy arrays. Then we can utilize array broadcasting
        # to evalute the integrals over products of primitive basis functions.
        result_lines = [f"numpy.sum({line})" for line in result_lines]
        # Drop ncomponents for simple integrals, as the python code can deal with contracted
        # GTOs via array broadcasting.
        if functions.ncomponents == 1:
            shape = shape[1:]
            shape_iter = [shape[1:] for shape in shape_iter]
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

    def render_func_dict(self, name, rendered_funcs):
        tpl = Template(
            """
            {{ name }} = {
            {% for func in rendered_funcs %}
            {{ func.Ls }}: {{ func.name }},
            {% endfor %}
            }
            """,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        rendered = textwrap.dedent(tpl.render(name=name, rendered_funcs=rendered_funcs)).strip()
        return rendered

    def render_module(self, functions, rendered_funcs):
        func_dict = self.render_func_dict(functions.name, rendered_funcs)
        tpl = Template(textwrap.dedent("""
            \"\"\"
            {{ header }}
            \"\"\"

            {{ comment }}

            import numpy
            {% if boys %}
            from pysisyphus.wavefunction.ints.boys import boys
            {% endif %}

            {% for ai in add_imports  %}
            {{ ai }}
            {% endfor %}

            {% for func in funcs %}
            {{ func.text }}
            {% endfor %}

            {{ func_dict }}
        """),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        rendered = tpl.render(
            header=functions.header,
            comment=functions.comment,
            boys=functions.boys,
            funcs=rendered_funcs,
            func_dict=func_dict,
        )
        rendered = textwrap.dedent(rendered)
        try:
            import black

            try:
                rendered = black.format_str(
                    rendered, mode=black.FileMode(line_length=90)
                )
                print("\t ... formatted Python code with black")
            except black.parsing.InvalidInput:
                print("Error while parsing with black. Dumping nontheless.")
        except ModuleNotFoundError:
            print("Skipped formatting with black, as it is not installed!")
        return rendered

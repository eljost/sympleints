from sympy.codegen.ast import Assignment
from sympy.printing.numpy import NumPyPrinter

from sympleints.Renderer import Renderer


class PythonRenderer(Renderer):
    ext = ".py"
    language = "Python"

    _tpls = {
        "func": "py_func.tpl",
        "func_dict": "py_func_dict.tpl",
        "module": "py_module.tpl",
    }
    _primitive = False
    _drop_dim = True

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
        # Here, we expect the orbital exponents and the contraction coefficients
        # to be 2d/3d/... numpy arrays. Then we can utilize array broadcasting
        # to evalute the integrals over products of primitive basis functions.
        if (not self._primitive) and (not functions.primitive):
            result_lines = [f"numpy.sum({line})" for line in result_lines]
        # Drop ncomponents for simple integrals, as the python code can deal with
        # contracted GTOs via array broadcasting.
        if self._drop_dim and functions.ncomponents == 1:
            shape = shape[1:]
            shape_iter = [shape[1:] for shape in shape_iter]
        results_iter = zip(shape_iter, result_lines)

        tpl = self.get_template(key="func")
        rendered = tpl.render(
            name=name,
            args=args,
            py_lines=py_lines,
            results_iter=results_iter,
            n_return_vals=len(reduced),
            doc_str=doc_str,
            shape=shape,
        )
        return rendered

    def render_func_dict(self, name, rendered_funcs):
        tpl = self.get_template(key="func_dict")
        rendered = tpl.render(name=name, rendered_funcs=rendered_funcs)
        return rendered

    def render_module(self, functions, rendered_funcs, **tpl_kwargs):
        func_dict = self.render_func_dict(functions.name, rendered_funcs)
        tpl = self.get_template(key="module")
        _tpl_kwargs = {
            "header": functions.header,
            "comment": functions.comment,
            "boys": functions.boys,
            "funcs": rendered_funcs,
            "func_dict": func_dict,
        }
        _tpl_kwargs.update(tpl_kwargs)
        rendered = tpl.render(**_tpl_kwargs)
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

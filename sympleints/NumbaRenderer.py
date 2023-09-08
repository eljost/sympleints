from sympleints.PythonRenderer import PythonRenderer
from sympleints.Functions import ArgKind


_type_map = {
    ArgKind.EXPO: "f8",
    ArgKind.CONTR: "f8",
    ArgKind.CENTER: "f8[:]",
    ArgKind.RESULT2: "f8[:, :]",
    ArgKind.RESULT3: "f8[:, :, :]",
    ArgKind.RESULT4: "f8[:, :, :, :]",
}

# TODO: investigate contiguity by using the ::1
# See: https://numba.pydata.org/numba-doc/latest/reference/types.html

_container_type_map = {
    ArgKind.EXPO: "f8[:]",
    ArgKind.CONTR: "f8[:]",
    ArgKind.CENTER: "f8[:]",
}


def func_type_from_functions(functions):
    """Numba function signature."""
    args = [_type_map[arg_kind] for arg_kind in functions.full_arg_kinds]
    # Add result type
    args += [_type_map[functions.result_kind]]
    args_str = ", ".join(args)
    signature = f"numba.types.void({args_str})"
    return signature


def driver_func_type_from_functions(functions):
    """Numba driver function signature w/ container types."""
    args = [_container_type_map[arg_kind] for arg_kind in functions.full_arg_kinds]
    # Prepend angular momenta that will be passed to the driver
    args = ["i8" for _ in range(functions.nbfs)] + args
    args_str = ", ".join(args)
    result = _type_map[functions.result_kind]
    signature = f"{result}({args_str}, func_dict_type)"
    return signature


class NumbaRenderer(PythonRenderer):
    ext = ".py"
    language = "Numba"

    _tpls = {
        "func": "numba_func.tpl",
        "func_dict": "py_func_dict.tpl",
        "driver": "numba_driver.tpl",
        "module": "numba_module.tpl",
    }
    _suffix = "_numba"
    _primitive = True

    def render_driver_func(self, functions, rendered_funcs):
        # Render func_dict inside driver
        tpl = self.get_template(key="driver")
        func_dict = self.render_func_dict("func", rendered_funcs)
        Ls = functions.Ls
        Ls_args = ", ".join(Ls)
        args = functions.full_args
        args_str = ", ".join(args)
        args = functions.full_container_args
        container_args_str = ", ".join(args)
        Ls_tuple = "(" + ",".join(Ls) + ")"
        exp_strs = map(str, functions.exponents)
        # func_ndim = ", ".join([":"] * functions.nbfs)
        driver_func_type = driver_func_type_from_functions(functions)
        rendered = tpl.render(
            # func_ndim=func_ndim,
            driver_func_type=driver_func_type,
            rendered_funcs=rendered_funcs,
            name=functions.name,
            exponents=exp_strs,
            Ls_args=Ls_args,
            container_args=container_args_str,
            args=args_str,
            Ls_tuple=Ls_tuple,
            func_dict=func_dict,
            nbfs=functions.nbfs,
        )
        return rendered

    def render_module(self, functions, rendered_funcs, **tpl_kwargs):
        func_type = func_type_from_functions(functions)
        driver_func = self.render_driver_func(functions, rendered_funcs)
        tpl_kwargs.update(
            {
                "func_type": func_type,
                "driver_func": driver_func,
            }
        )
        return super().render_module(functions, rendered_funcs, **tpl_kwargs)

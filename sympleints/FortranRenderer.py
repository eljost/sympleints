import os
import re
import tempfile
import warnings

from sympy.codegen.ast import Assignment
from sympy.printing.fortran import FCodePrinter

from sympleints.Renderer import Renderer
from sympleints.helpers import shell_shape_iter


def format_with_fprettify(fortran: str):
    try:
        import fprettify
    except ModuleNotFoundError:
        print("Skipped formatting with fprettify, as it is not installed!")
        return fortran
    except (AttributeError, ValueError):
        # pytest patches sys.stdin, on which fprettify tries to call detach() on
        # which fails, as DontReadFromInput does not have this method.
        return fortran

    # I wonder if fprettify can also deal with strings instead of files only?!
    # I did not manage to find out ... so we use some temporary files here.
    try:
        fp = tempfile.NamedTemporaryFile("w", delete=False)
        fp.write(fortran)
        fp.close()
        fprettify.reformat_inplace(fp.name)
        with open(fp.name) as handle:
            fortran_formatted = handle.read()
        os.remove(fp.name)
        print("\t ... formatted Fortran code with fprettify")
    except fprettify.FprettifyException:
        print("Error while running fprettify. Dumping nontheless.")
        fortran_formatted = fortran
    return fortran_formatted


class FCodePrinterMod(FCodePrinter):
    boys_re = re.compile(r"boys\(([d\d\.]+),(.+)")

    def _print_Function(self, expr):
        func = super()._print_Function(expr)
        # Sympy prints everything as float (1.0d0, 2.0d0 etc.), even integers, but the
        # first argument 'n' to the Boys function must be integer.
        if func.startswith("boys"):
            # Only try to fix calls to the Boys function when a double is present
            if mobj := self.boys_re.match(func):
                as_float = float(mobj.group(1).lower().replace("d", "e"))
                as_int = int(as_float)
                assert abs(as_float - as_int) <= 1e-14
                remainder = mobj.group(2)
                func = f"boys({as_int},{remainder}"
        return func

    def _print_Indexed(self, expr):
        # prints I[0] as I[1], i.e., increments the index by one.
        inds = [self._print(i + 1) for i in expr.indices]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))

    def _print_IndexedSlice(self, expr):
        start = expr.start_index + 1
        # end_index is exclusive; Fortran end indices are inclusive.
        stop = expr.end_index + 1
        return f"{expr.org_name}({start}:{stop})"


def get_fortran_print_func(**print_settings):
    # This allows using the 'boys' function without producing an error
    _print_settings = {
        "allow_unknown_functions": True,
        # Without disabling contract some expressions will raise ValueError.
        "contract": False,
        "standard": 2008,
        "source_format": "free",
    }
    _print_settings.update(print_settings)
    print_func = FCodePrinterMod(_print_settings).doprint
    return print_func


def make_fortran_comment(comment_str):
    return "".join([f"! {line}\n" for line in comment_str.strip().split("\n")])


class FortranRenderer(Renderer):
    ext = ".f90"
    real_kind = "kind=real64"
    res_name = "res"
    language = "Fortran"

    def shell_shape_iter(self, *args, **kwargs):
        # Start indexing at 1, instead of 0.
        return shell_shape_iter(*args, start_at=1, **kwargs)

    def get_argument_declaration(self, functions, contracted=False):
        tpl = self.get_template(fn="fortran_arg_declaration.tpl")
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
        if functions.primitive:
            warnings.warn("primitive=True is currently ignored by FortranRenderer!")

        print_func = get_fortran_print_func()
        assignments = [Assignment(lhs, rhs) for lhs, rhs in repls]
        repl_lines = [print_func(as_) for as_ in assignments]
        results = [print_func(red) for red in reduced]
        res_len = len(reduced)
        results_iter = zip(shape_iter, results)

        doc_str = make_fortran_comment(doc_str)
        arg_declaration = self.get_argument_declaration(functions)

        tpl = self.get_template(fn="fortran_function.tpl")
        rendered = tpl.render(
            name=name,
            args=functions.full_args,
            doc_str=doc_str,
            arg_declaration=arg_declaration,
            res_name=self.res_name,
            res_len=res_len,
            assignments=assignments,
            repl_lines=repl_lines,
            results_iter=results_iter,
            reduced=reduced,
            kind=self.real_kind,
        )
        return rendered

    """
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

        tmps = [lhs for lhs, rhs in repls]
        from sympy import IndexedBase
        tmp = IndexedBase('tmp', shape=len(tmps))
        map_ = {}
        for i, rhs_ in enumerate(tmps, 1):
            map_[rhs_] = tmp[i - 1]
        rhs_subs = list()
        for _, rhs in repls:
            rhs_subs.append(rhs.subs(map_))
        red_subs = list()
        for red in reduced:
            red_subs.append(red.subs(map_))
        results_sub = [print_func(red) for red in red_subs]
        results_iter = zip(shape_iter, results_sub)
        repl_rhss = [print_func(rhs) for rhs in rhs_subs]
        doc_str = make_fortran_comment(doc_str)
        arg_declaration = self.get_argument_declaration(functions)

        tpl = self.get_template(fn="fortran_function_arr.tpl")
        rendered = tpl.render(
                name=name,
                args=functions.full_args,
                doc_str=doc_str,
                arg_declaration=arg_declaration,
                res_name=self.res_name,
                res_len=res_len,
                assignments=assignments,
                repl_lines=repl_lines,
                results_iter=results_iter,
                reduced=reduced,
                kind=self.real_kind,
                repl_rhs=repl_rhss,
            )
        return rendered
    """

    def render_f_init(self, name, rendered_funcs, func_array_name="func_array"):
        tpl = self.get_template(fn="fortran_init.tpl")
        f_init = tpl.render(
            name=name,
            funcs=rendered_funcs,
            func_array_name=func_array_name,
        )
        return f_init

    def render_contracted_driver(self, functions):
        tpl = self.get_template(fn="fortran_contracted_driver.tpl")
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

        tpl = self.get_template(fn="fortran_module.tpl")
        rendered = tpl.render(
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
        rendered = format_with_fprettify(rendered)
        return rendered

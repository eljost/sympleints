import abc
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Tuple

from jinja2 import Environment, PackageLoader, select_autoescape

from sympleints.Functions import Functions
from sympleints.helpers import func_name_from_Ls, shell_shape, shell_shape_iter


@dataclass
class RenderedFunction:
    name: str
    Ls: Tuple[int]
    text: str


class Renderer(abc.ABC):
    _tpls = {}
    _suffix = ""
    # Whether the functions/subroutines deal with primitive or contracted Gaussians
    _primitive = True

    env = Environment(
        loader=PackageLoader("sympleints"),
        # We proably don't need autoescape...
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    def get_template(self, *, key=None, fn=None):
        assert (key is not None) or (
            fn is not None
        ), "Either 'key' or 'fn' must be provided!"
        if key is not None:
            fn = self._tpls[key]
        tpl = self.env.get_template(fn)
        return tpl

    def shell_shape_iter(self, *args, **kwargs):
        return shell_shape_iter(*args, **kwargs)

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def render_equi_function(
        self,
        functions,
        name,
        act_name,
        equi_inds,
        L_tots,
        from_axes,
        to_axes,
    ):
        raise NotImplementedError

    def render_functions(self, functions: Functions):
        args = functions.full_args
        ncomponents = functions.ncomponents
        if len(hermi_inds := functions.hermitian) >= 2:
            hermi_inds = functions.hermitian
            assert len(hermi_inds) == 2
            org_inds = hermi_inds
            equi_inds = org_inds.copy()
            hi, hj = hermi_inds
            equi_inds[hi], equi_inds[hj] = equi_inds[hj], equi_inds[hi]
        else:
            equi_inds = []

        rendered_funcs = list()
        for L_tots, (repls, reduced) in functions.ls_exprs:
            shape = shell_shape(
                L_tots, ncomponents=functions.ncomponents, cartesian=functions.cartesian
            )
            shape_iter = self.shell_shape_iter(
                L_tots, ncomponents=ncomponents, cartesian=functions.cartesian
            )
            try:
                doc_str = functions.doc_func(L_tots) + "\n\n"
            except TypeError:
                doc_str = ""
            doc_str += "Generated code; DO NOT modify by hand!"
            name = func_name_from_Ls(functions.name, L_tots)
            print(f"Rendering {self.language} code for '{name}' ... ", end="")
            start = time.time()
            func = self.render_function(
                functions,
                repls,
                reduced,
                shape,
                shape_iter,
                args=args,
                name=name,
                doc_str=doc_str,
            )
            dur = time.time() - start
            print(f"finished in {dur: >8.2f} s")
            rendered_funcs.append(RenderedFunction(name=name, Ls=L_tots, text=func))

            # Request generation of equivalent integrals here
            if len(equi_inds) > 0 and (L_tots[equi_inds[1]] > L_tots[equi_inds[0]]):
                assert len(equi_inds) == 2
                L_tots_equi = list(L_tots)
                i, j = equi_inds
                L_tots_equi[i], L_tots_equi[j] = L_tots_equi[j], L_tots_equi[i]
                L_tots_equi = tuple(L_tots_equi)
                name_equi = func_name_from_Ls(functions.name, L_tots_equi)

                from_axes = tuple(equi_inds)
                to_axes = tuple(org_inds)

                nbfs = functions.nbfs
                # Axes/indices are missing
                if len(from_axes) != nbfs:
                    to_axes = tuple(range(nbfs))
                    from_axes = list(range(nbfs))
                    from_axes[hi], from_axes[hj] = from_axes[hj], from_axes[hi]
                    from_axes = tuple(from_axes)

                more_components = functions.ncomponents > 1
                add_axis = more_components or (not self._drop_dim and nbfs == 2)
                if add_axis:
                    from_axes = tuple([0] + [fa + 1 for fa in from_axes])
                    to_axes = tuple([0] + [ta + 1 for ta in to_axes])

                reshape = shape if functions.ncomponents > 1 else shape[1:]

                func_equi = self.render_equi_function(
                    functions,
                    name,
                    name_equi,
                    equi_inds,
                    reshape,
                    from_axes,
                    to_axes,
                )
                rendered_funcs.append(
                    RenderedFunction(name=name_equi, Ls=L_tots_equi, text=func_equi)
                )

        return rendered_funcs

    @abc.abstractmethod
    def render_module(self, functions, rendered_funcs, **tpl_kwargs):
        raise NotImplementedError

    def render(self, functions: Functions):
        """Main driver to render Functions object to actual code."""
        rendered_funcs = self.render_functions(functions)
        module = self.render_module(
            functions,
            rendered_funcs,
        )
        return module

    def write(self, out_dir, name, text):
        fn = (out_dir / name).with_suffix(self.ext)
        with open(fn, "w") as handle:
            handle.write(text)
        return fn

    def render_write(self, functions: Functions, out_dir: Path):
        module = self.render(functions)
        # out_name = functions.name + self._suffix + self.ext
        out_name = functions.name + self.ext
        fn = self.write(out_dir, out_name, module)
        return fn

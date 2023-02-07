import abc
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Tuple

from sympleints.Functions import Functions
from sympleints.helpers import func_name_from_Ls, shell_shape, shell_shape_iter


@dataclass
class RenderedFunction:
    name: str
    Ls: Tuple[int]
    text: str


class Renderer(abc.ABC):
    def shell_shape_iter(self, *args, **kwargs):
        return shell_shape_iter(*args, **kwargs)

    @abc.abstractmethod
    def render_function(
        # self, repls, reduced, shape, shape_iter, args, name, doc_str="", **kwargs
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
        pass

    def render_functions(self, functions: Functions):
        # ls_exprs, args, base_name, doc_func, add_imports=None):

        args = functions.full_args
        ncomponents = functions.ncomponents

        rendered_funcs = list()
        # func_map = list()
        for L_tots, (repls, reduced) in functions.ls_exprs:
            shape = shell_shape(L_tots, cartesian=True)
            shape_iter = self.shell_shape_iter(
                L_tots, ncomponents=ncomponents, cartesian=True
            )
            try:
                doc_str = functions.doc_func(L_tots) + "\n\n"
            except AttributeError:
                doc_str = ""
            doc_str += "Generated code; DO NOT modify by hand!"
            name = func_name_from_Ls(functions.name, L_tots)
            # func_map.append((L_tots, name))
            print(f"Rendering '{name}' ... ", end="")
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
        return rendered_funcs

    @abc.abstractmethod
    def render_module(self, functions, rendered_funcs):
        pass

    def render(self, functions: Functions):
        rendered_funcs = self.render_functions(functions)
        module = self.render_module(
            functions,
            rendered_funcs,
        )
        #
        return module

    def write(self, out_dir, name, text):
        fn = (out_dir / name).with_suffix(self.ext)
        with open(fn, "w") as handle:
            handle.write(text)
        return fn

    def render_write(self, functions: Functions, out_dir: Path):
        module = self.render(functions)
        fn = self.write(out_dir, functions.name + self.ext, module)
        return fn

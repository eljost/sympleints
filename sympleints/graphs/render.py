from sympleints.graphs.generate import GeneratedIntegral
from sympleints.FortranRenderer import format_with_fprettify, get_fortran_print_func

from jinja2 import Environment, PackageLoader, select_autoescape


ENV = Environment(
    loader=PackageLoader("sympleints.graphs"),
    # We proably don't need autoescape...
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_fortran(gen_integral: GeneratedIntegral):
    gi = gen_integral  # Shortcut

    print_func = get_fortran_print_func()
    template = ENV.get_template(f"{gi.integral.name}_func.tpl")

    blocks = dict()
    for block_name, cur_assignments in gi.assignments.items():
        lines = [print_func(ass) for ass in cur_assignments]
        blocks[block_name] = lines
    rendered = template.render(
        name=gi.name,
        blocks=blocks,
        array_defs=gi.array_defs,
        shell_size=gi.shell_size,
        target_array_name=gi.target_array_name,
    )
    # Better do this in the module?!
    # rendered = format_with_fprettify(rendered)
    return rendered


def render_fortran_equi(gen_integral: GeneratedIntegral):
    gi = gen_integral  # Shortcut
    ncenters = gi.integral.ncenters

    template = ENV.get_template(f"int_equi_func_{ncenters}c.tpl")
    rendered = template.render(
        name=gi.name,
        act_name=gi.act_genint.name,
        act_size=gi.act_genint.integral.shell_size,
        act_shape=gi.act_genint.integral.shell_shape,
    )
    return rendered


def render_fortran_integral(
    gen_integral: GeneratedIntegral,
) -> tuple[list[str], list[tuple[int, ...]]]:
    rendered_funcs = list()
    L_tots = list()
    rendered_funcs.append(render_fortran(gen_integral))
    L_tots.append(gen_integral.L_tots)
    # Also create equivalent integrals
    for equi_genint in gen_integral.generate_equivalent():
        rendered_funcs.append(render_fortran_equi(equi_genint))
        L_tots.append(equi_genint.L_tots)
    return rendered_funcs, L_tots


def render_fortran_module_from_rendered(
    name: str, lmax: int, lauxmax: int, funcs: list[str], L_tots: list[tuple[int]]
) -> str:
    tpl = ENV.get_template(f"{name}_mod.tpl")
    rendered = tpl.render(
        integral_name=name,
        lmax=lmax,  # TODO
        lauxmax=lauxmax,  # TODO
        funcs=funcs,
        L_tots=L_tots,
    )
    # Sadly, quite expensive for higher angular momenta.
    # rendered = format_with_fprettify(rendered)
    return rendered

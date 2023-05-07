from sympleints.FortranRenderer import format_with_fprettify, get_fortran_print_func

from jinja2 import Environment, PackageLoader, select_autoescape


ENV = Environment(
    loader=PackageLoader("sympleints.graphs"),
    # We proably don't need autoescape...
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_fortran(gen_integral):
    gi = gen_integral  # Shortcut

    print_func = get_fortran_print_func()
    name = gi.integral.name
    template = ENV.get_template(f"{name}_func.tpl")

    blocks = dict()
    for block_name, cur_assignments in gi.assignments.items():
        lines = [print_func(ass) for ass in cur_assignments]
        blocks[block_name] = lines
    rendered = template.render(
        name=f"{name}_{gi.key}",
        blocks=blocks,
        array_defs=gi.array_defs,
        shell_size=gi.shell_size,
        target_array_name=gi.target_array_name,
    )
    # Better do this in the module?!
    # rendered = format_with_fprettify(rendered)
    return rendered


def render_fortran_module(gen_integrals, lmax, lauxmax):
    gi0 = gen_integrals[0]
    funcs = [render_fortran(gi) for gi in gen_integrals]
    L_tots = [gi.L_tots for gi in gen_integrals]
    tpl = ENV.get_template(f"{gi0.integral.name}_mod.tpl")
    rendered = tpl.render(
        integral_name=gi0.integral.name,
        lmax=lmax,  # TODO
        lauxmax=lauxmax,  # TODO
        funcs=funcs,
        L_tots=L_tots,
    )
    rendered = format_with_fprettify(rendered)
    return rendered

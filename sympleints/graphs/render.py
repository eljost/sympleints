from sympleints.FortranRenderer import format_with_fprettify, get_fortran_print_func

from jinja2 import Environment, PackageLoader, select_autoescape


def render_fortran(gen_integral):
    gi = gen_integral  # Shortcut

    print_func = get_fortran_print_func()
    env = Environment(
        loader=PackageLoader("sympleints.graphs"),
        # We proably don't need autoescape...
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    name = gi.integral.name
    template = env.get_template(f"{name}_func.tpl")

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
    rendered = format_with_fprettify(rendered)
    return rendered

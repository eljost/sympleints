int_tuple_type = numba.types.UniTuple(i8, {{ nbfs }})
func_dict_type = numba.types.DictType(int_tuple_type, func_type)

# Sadly, this function can't be cached.
@numba.jit(func_dict_type(), nopython=True, cache=True)
def get_func_dict():
    # Can we somehow utilize the 'func_dict_type' definition above?!
    #
    # This definition below does not work but leads to strange errors.
    # func_dict = numba.typed.Dict.empty(
    # key_type=func_dict_type.key_type,
    # value_type=func_dict_type.value_type,
    # )
    # numba plzzz
    func_dict = numba.typed.Dict.empty(
        key_type=int_tuple_type,
        value_type=func_type,
    )
    {% for func in rendered_funcs %}
    func_dict[{{ func.Ls }}] = {{ func.name }}
    {% endfor %}
    return func_dict


{#
driver_func_type = numba.types.FunctionType(
    {{ driver_func_type }}
)

@numba.jit(
    driver_func_type.signature,
    nopython=True,
    nogil=True,
)
def {{ name }}({{ Ls_args }}, {{ container_args }}, func_dict):

    # TODO: generalize to spherical bfs?
    shape = [(L + 2) * (L + 1) // 2 for L in {{ Ls_tuple }}]
    {% if nbfs == 1 %}
    result = numpy.zeros(shape[0])
    {% elif nbfs == 2 %}
    result = numpy.zeros((shape[0], shape[1]))
    {% elif nbfs == 3 %}
    result = numpy.zeros((shape[0], shape[1], shape[2]))
    {% endif %}
    {% for exp in exponents %}
    n{{ exp[0] }} = {{ exp }}s.size
    {% endfor %}
    func = func_dict[{{ Ls_tuple }}]
    
    {% if nbfs == 1 %}
    for i in range(na):
        ax = axs[i]
        da = das[i]
        func({{ args }}, result)
	{% elif nbfs == 2 %}
    for i in range(na):
        ax = axs[i]
        da = das[i]
        for j in range(nb):
            bx = bxs[j]
            db = dbs[j]
            func({{ args }}, result)
	{% elif nbfs == 3 %}
    for i in range(na):
        ax = axs[i]
        da = das[i]
        for j in range(nb):
            bx = bxs[j]
            db = dbs[j]
            for j in range(nc):
                cx = cxs[j]
                dc = dcs[j]
                func({{ args }}, result)
	{% endif %}
    return result
#}

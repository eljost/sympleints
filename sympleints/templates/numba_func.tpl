@numba.jit(
    func_type.signature,
    nopython=True,
    nogil=True,
    fastmath=True,
)
def {{ name }}({{ args }}, result):
    {% if doc_str %}
    """{{ doc_str }}"""
    {% endif %}

    {% for line in py_lines %}
    {{ line }}
    {% endfor %}

    # {{ n_return_vals }} item(s)
    {% for inds, res_line in results_iter %}
    result[{{ inds|join(", ")}}] += {{ res_line }}
    {% endfor %}

def {{ name }}({{ args }}):
    {% if doc_str %}
    """{{ doc_str }}"""
    {% endif %}

    result = numpy.zeros({{ shape }}, dtype=float)

    {% for line in py_lines %}
    {{ line }}
    {% endfor %}

    # {{ n_return_vals }} item(s)
    {% for inds, res_line in results_iter %}
    result[{{ inds|join(", ")}}] = {{ res_line }}
    {% endfor %}
    return result

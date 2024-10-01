def {{ name }}({{ args }}, result):
    {% if doc_str %}
    """{{ doc_str }}"""
    {% endif %}

    {# result = numpy.zeros({{ shape }}, dtype=float) #}

    {% for line in py_lines %}
    {{ line }}
    {% endfor %}

    # {{ n_return_vals }} item(s)
    # TODO: check if removing the += gives problems w/ primitive=True
    {% for _, res_line in results_iter %}
    result[{{ loop.index0 }}] = {{ res_line }}
    {% endfor %}

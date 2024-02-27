subroutine {{ name }} ({{ args|join(", ") }}, {{ res_name }})

{% if doc_str %}
{{ doc_str }}
{% endif %}

{{ arg_declaration }}

! Intermediate quantities
{% for as_ in assignments %}
real({{ kind }}) :: {{ as_.lhs }}
{% endfor %}

{% for line in repl_lines %}
{{ line }}
{% endfor %}

{% for inds, res_line in results_iter %}
{{ res_name }}({{ loop.index }}) = {{ res_line }}
{% endfor %}
end subroutine {{ name }}

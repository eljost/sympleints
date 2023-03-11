subroutine {{ name }} ({{ args|join(", ") }}, {{ res_name }})

{% if doc_str %}
{{ doc_str }}
{% endif %}

{{ arg_declaration }}

! Intermediate quantities
real({{ kind }}) :: tmp({{ assignments|length }})

{% for rhs in repl_rhs %}
tmp({{ loop.index }}) = {{ rhs }}
{% endfor %}

{% for inds, res_line in results_iter %}
{{ res_name }}({{ inds|join(", ") }})  = {{ res_line }}
{% endfor %}
end subroutine {{ name }}

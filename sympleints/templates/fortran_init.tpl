subroutine {{ name }}_init ()
! Initializer procedure. MUST be called before {{ name }} can be called.
{% for func in funcs %}
    {{ func_array_name }}({{ func.Ls|join(", ") }})%f => {{ func.name }}
{% endfor %}
end subroutine

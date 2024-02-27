subroutine {{ equi_name }} ({{ args|join(", ") }}, {{ res_name }})

  ! See docstring of {{ name }}.

{{ arg_declaration }}

call {{ name }}({{ equi_args|join(", ") }}, {{ res_name }})
call {{ resort_func }}({{ res_name }}, {{ sizes|join(", ")}}, {{ ncomponents }})
end subroutine {{ equi_name }}

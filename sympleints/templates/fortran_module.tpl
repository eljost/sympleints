{{ header }}


module {{ mod_name }}

use iso_fortran_env, only: real64
{% if boys %}
use mod_boys, only: boys
{% endif %}

implicit none

type fp
    procedure({{ interface_name }}) ,pointer ,nopass :: f =>null()
end type fp

interface
    subroutine {{ interface_name }}({{ args|join(", ")}})
      import :: real64
      {{ arg_declaration }}
    end subroutine {{ interface_name }}
end interface

type(fp) :: func_array({{ func_arr_dims|join(", ") }})

contains
{{ init }}

{{ contr_driver }}

{% for func in funcs %}
    {{ func.text }}
{% endfor %}

end module {{ mod_name }}

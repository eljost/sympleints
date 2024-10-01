submodule (mod_pa_{{ integral_name }}) mod_pa_{{ submodule_name }}
  use mod_pa_constants, only: dp, PI
  use mod_pa_boys, only: boys

  implicit none

contains
  {% for func in funcs %}
    {{ func }}

  {% endfor %}
end submodule mod_pa_{{ submodule_name }}

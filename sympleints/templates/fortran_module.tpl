{{ header }}


module {{ mod_name }}

use iso_fortran_env, only: real64
{% if boys %}
use mod_boys, only: boys
{% endif %}

implicit none

type fp
    procedure({{ interface_name }}), pointer, nopass :: f =>null()
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

{% if resort_func == "resort_ba_ab" %}
   subroutine resort_ba_ab({{ res_name }}, sizea, sizeb, ncomponents)
      real(real64), intent(in out) :: {{ res_name }}(:)
      integer, intent(in) :: sizea, sizeb, ncomponents
      integer :: sizeab, component_size
      integer :: nc, a, b, i, j
      real(real64) :: tmp(size({{ res_name }}))

      sizeab = sizea * sizeb
      i = 1
      do nc = 1, ncomponents
         do b = 1, sizeb
            do a = 1, sizea
               j = component_size + ((a-1) * sizeb) + b
               tmp(j) = {{ res_name }}(i)
               i = i + 1
            end do
         end do
         component_size = component_size + sizeab
      end do

      {{ res_name }} = tmp
   end subroutine resort_ba_ab
{% endif %}

{{ contr_driver }}

{% for func in funcs %}
    {{ func.text }}
{% endfor %}

end module {{ mod_name }}

subroutine {{ name }}({{ L_args|join(", ") }}, {{ args|join(", ") }})
    integer, intent(in) :: {{ Ls|join(", ") }}
    {{ arg_declaration }}
    real({{ kind }}), allocatable :: res_tmp(:)
    ! Initializing with => null () adds an implicit save, which will mess
    ! everything up when running with OpenMP.
    procedure({{ name }}_proc), pointer :: fncpntr
    integer :: {{ loop_counter|join(", ") }}

    allocate(res_tmp, mold={{ res_name }})
    fncpntr => func_array({{ L_args|join(", ") }})%f

    {{ res_name }} = 0
    {{ loops|join("\n") }}
        call fncpntr({{ pntr_args }}, res_tmp)
        {{ res_name }} = {{ res_name }} + res_tmp
    {% for _ in loops %}
    end do
    {% endfor %}
    deallocate(res_tmp)
end subroutine {{ name }}

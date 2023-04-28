subroutine {{ name }}({{ L_args|join(", ") }}, {{ args|join(", ") }})
    integer, intent(in) :: {{ Ls|join(", ") }}
    {{ arg_declaration }}
    real(kind=real64), allocatable, dimension({{ res_dim }}) :: res_tmp
    ! Initializing with => null () adds an implicit save, which will mess
    ! everything up when running with OpenMP.
    procedure({{ name }}_proc), pointer :: fncpntr
    integer :: {{ loop_counter|join(", ") }}

    allocate(res_tmp, mold=res)
    fncpntr => func_array({{ L_args|join(", ") }})%f

    res = 0
    {{ loops|join("\n") }}
        call fncpntr({{ pntr_args }}, res_tmp)
        res = res + res_tmp
    {% for _ in loops %}
    end do
    {% endfor %}
    deallocate(res_tmp)
end subroutine {{ name }}

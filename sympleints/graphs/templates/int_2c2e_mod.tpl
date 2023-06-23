module mod_{{ integral_name }}
  use iso_fortran_env, only: real64

  use mod_constants, only: PI
  use mod_boys, only: boys
  
  implicit none

  type fp
      procedure({{ integral_name }}_proc), pointer, nopass :: f => null()
  end type fp

  interface
    subroutine {{ integral_name }}_proc(axs, das, A, bxs, dbs, B, R, ress)
      import :: real64

      ! Orbital exponents
      real(kind=real64), intent(in) :: axs(:), bxs(:)
      ! Contraction coefficients
      real(kind=real64), intent(in) :: das(:), dbs(:)
      ! Centers
      real(kind=real64), intent(in) :: A(3), B(3), R(3)
      real(kind=real64), intent(out) :: ress(:)
      end subroutine {{ integral_name }}_proc
   end interface

   type(fp) :: func_array(0:{{ lauxmax }}, 0:{{ lauxmax }})

contains
  subroutine {{ integral_name }}_init()
    ! Initializer procedure. MUST be called before {{ integral_name }} can be called.
    {% for L_tot in L_tots %}
        func_array({{ L_tot|join(", ") }})%f => {{ integral_name }}_{{ L_tot|join("") }}
    {% endfor %}
  end subroutine {{ integral_name }}_init

  subroutine {{ integral_name }}(La, Lb, axs, das, A, bxs, dbs, B, R, res)
      integer, intent(in) :: La, Lb
      ! Orbital exponents
      real(kind=real64), intent(in) :: axs(:), bxs(:)
      ! Contraction coefficients
      real(kind=real64), intent(in) :: das(:), dbs(:)
      ! Centers
      real(kind=real64), intent(in) :: A(3), B(3), R(3)
      real(kind=real64), intent(out) :: res(:, :)

      real(kind=real64), allocatable :: ress(:)
      ! Initializing with => null () adds an implicit save, which will mess
      ! everything up when running with OpenMP.
      procedure({{ integral_name }}_proc), pointer :: fncpntr

      allocate(ress(size(res)))
      fncpntr => func_array(La, Lb)%f

      call fncpntr(axs, das, A, bxs, dbs, B, R, ress)

      res(:, :) = reshape(ress, (/2*La+1, 2*Lb+1/))
      if (La < Lb) then
        res(:, :) = reshape(ress, (/2*La+1, 2*Lb+1/), order=(/ 1, 2 /))
      else
        res(:, :) = reshape(ress, (/2*La+1, 2*Lb+1/), order=(/ 2, 1 /))
      end if
  end subroutine {{ key }}
  
  {% for func in funcs %}
    {{ func }}
    
  {% endfor %}
end module mod_{{ integral_name }}

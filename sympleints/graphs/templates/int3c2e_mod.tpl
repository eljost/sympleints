module mod_{{ integral_name }}
  use iso_fortran_env, only: real64

  use mod_constants, only: PI, PI_4
  use mod_boys, only: boys
  
  implicit none

  real(kind=real64), parameter :: epsilon = 1d-10
  real(kind=real64), parameter :: epsilon2 = epsilon**2

  type fp
      procedure({{ integral_name }}_proc), pointer, nopass :: f => null()
  end type fp

  interface
    subroutine {{ integral_name }}_proc(axs, das, A, bxs, dbs, B, cxs, dcs, C, ress)
      import :: real64

      ! Orbital exponents
      real(kind=real64), intent(in) :: axs(:), bxs(:), cxs(:)
      ! Contraction coefficients
      real(kind=real64), intent(in) :: das(:), dbs(:), dcs(:)
      ! Centers
      real(kind=real64), intent(in), dimension(3) :: A, B, C
      real(kind=real64), intent(out) :: ress(:)
      end subroutine {{ integral_name }}_proc
   end interface

   type(fp) :: func_array(0:{{ lmax }}, 0:{{ lmax }}, 0:{{ lauxmax }})

contains
  subroutine {{ integral_name }}_init()
    ! Initializer procedure. MUST be called before {{ integral_name }} can be called.
    {% for L_tot in L_tots %}
        func_array({{ L_tot|join(", ") }})%f => {{ integral_name }}_{{ L_tot|join("") }}
    {% endfor %}
  end subroutine {{ integral_name }}_init

  subroutine {{ integral_name }}(La, Lb, Lc, axs, das, A, bxs, dbs, B, cxs, dcs, C, res)
      integer, intent(in) :: La, Lb, Lc
      ! Orbital exponents
      real(kind=real64), intent(in) :: axs(:), bxs(:), cxs(:)
      ! Contraction coefficients
      real(kind=real64), intent(in) :: das(:), dbs(:), dcs(:)
      ! Centers
      real(kind=real64), intent(in), dimension(3) :: A, B, C
      real(kind=real64), intent(out) :: res(:, :, :)

      real(kind=real64), allocatable :: ress(:)
      ! Initializing with => null () adds an implicit save, which will mess
      ! everything up when running with OpenMP.
      procedure({{ integral_name }}_proc), pointer :: fncpntr

      allocate(ress(size(res)))
      fncpntr => func_array(La, Lb, Lc)%f

      call fncpntr(axs, das, A, bxs, dbs, B, cxs, dcs, C, ress)
      res(:, :, :) = reshape(ress, (/2*La+1, 2*Lb+1, 2*Lc+1/))
  end subroutine {{ key }}
  
  {% for func in funcs %}
    {{ func }}
  {% endfor %}
end module mod_{{ integral_name }}

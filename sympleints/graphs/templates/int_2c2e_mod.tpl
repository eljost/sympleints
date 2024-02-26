module mod_pa_{{ integral_name }}
  use mod_pa_constants, only: dp, PI
  use mod_pa_boys, only: boys
  
  implicit none

  type fp
      procedure({{ integral_name }}_proc), pointer, nopass :: f => null()
  end type fp

  interface
    subroutine {{ integral_name }}_proc(axs, das, A, bxs, dbs, B, res)
      import :: dp

      ! Orbital exponents
      real(dp), intent(in) :: axs(:), bxs(:)
      ! Contraction coefficients
      real(dp), intent(in) :: das(:), dbs(:)
      ! Centers
      real(dp), intent(in) :: A(3), B(3)
      real(dp), intent(out) :: res(:)
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
      real(dp), intent(in) :: axs(:), bxs(:)
      ! Contraction coefficients
      real(dp), intent(in) :: das(:), dbs(:)
      ! Centers
      real(dp), intent(in) :: A(3), B(3), R(3)
      real(dp), intent(out) :: res(:)
      
      ! Initializing with => null () adds an implicit save, which will mess
      ! everything up when running with OpenMP.
      procedure({{ integral_name }}_proc), pointer :: fncpntr

      fncpntr => func_array(La, Lb)%f

      call fncpntr(axs, das, A, bxs, dbs, B, res)

      ! With (La < Lb) the second index b will be the fastest changing index
      ! With (Lb < La) the first index a will be the fastest changing index
      ! which is more suitable for a Fortran loop-structure
      if (Lb < La) then
        res = pack(reshape(res, (/2*La+1, 2*Lb+1/), order=(/ 2, 1 /)), .true.)
      end if
  end subroutine {{ key }}
  
  {% for func in funcs %}
    {{ func }}
    
  {% endfor %}
end module mod_pa_{{ integral_name }}

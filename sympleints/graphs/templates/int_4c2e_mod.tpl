module mod_pa_{{ integral_name }}

  use mod_pa_constants, only: dp, i4, PI, PI_4
  use mod_pa_boys, only: boys
  
  implicit none

  type fp
      procedure({{ integral_name }}_proc), pointer, nopass :: f => null()
  end type fp

  interface
    subroutine {{ integral_name }}_proc(axs, das, A, bxs, dbs, B, cxs, dcs, C, dxs, dds, D, res)
      import :: dp

      ! Orbital exponents
      real(dp), intent(in) :: axs(:), bxs(:), cxs(:), dxs(:)
      ! Contraction coefficients
      real(dp), intent(in) :: das(:), dbs(:), dcs(:), dds(:)
      ! Centers
      real(dp), intent(in) :: A(3), B(3), C(3), D(3)
      real(dp), intent(out) :: res(:)
      end subroutine {{ integral_name }}_proc
   end interface

   type(fp) :: func_array(0:{{ lmax }}, 0:{{ lmax }}, 0:{{ lmax }}, 0:{{ lmax }})

contains
  subroutine {{ integral_name }}_init()
    ! Initializer procedure. MUST be called before {{ integral_name }} can be called.
    {% for L_tot in L_tots %}
        func_array({{ L_tot|join(", ") }})%f => {{ integral_name }}_{{ L_tot|join("") }}
    {% endfor %}
  end subroutine {{ integral_name }}_init

  subroutine {{ integral_name }}(La, Lb, Lc, Ld, axs, das, A, bxs, dbs, B, cxs, dcs, C, dxs, dds, D, res)
      integer(i4), intent(in) :: La, Lb, Lc, Ld
      ! Orbital exponents
      real(dp), intent(in) :: axs(:), bxs(:), cxs(:), dxs(:)
      ! Contraction coefficients
      real(dp), intent(in) :: das(:), dbs(:), dcs(:), dds(:)
      ! Centers
      real(dp), intent(in) :: A(3), B(3), C(3), D(3)
      real(dp), intent(out) :: res(:)

      ! Initializing with => null () adds an implicit save, which will mess
      ! everything up when running with OpenMP.
      procedure({{ integral_name }}_proc), pointer :: fncpntr

      fncpntr => func_array(La, Lb, Lc, Ld)%f

      call fncpntr(axs, das, A, bxs, dbs, B, cxs, dcs, C, dxs, dds, D, res)
      !if ((La < Lb) .and. (Ld < Lc)) then
      ! res(:, :, :, :) = reshape(ress, (/2*La+1, 2*Lb+1, 2*Lc+1, 2*Ld+1/), order=(/ 2, 1, 4, 3 /))
      !else
      !res(:, :, :, :) = reshape(ress, (/2*La+1, 2*Lb+1, 2*Lc+1, 2*Ld+1/))
      !end if
  end subroutine {{ integral_name }}

  subroutine int_schwarz (La, Lb, axs, das, A, bxs, dbs, B, R res)
      integer(i4), intent(in) :: La, Lb
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

      fncpntr => func_array(La, Lb, La, Lb)%f

      call fncpntr(axs, das, A, bxs, dbs, B, axs, das, A, bxs, dbs, B, res)
      !if (La < Lb) then
      !  res(:, :, :, :) = reshape(ress, (/2*La+1, 2*Lb+1, 2*La+1, 2*Lb+1/), order=(/ 2, 1, 4, 3 /))
      !else
      !res(:, :, :, :) = reshape(ress, (/2*La+1, 2*Lb+1, 2*La+1, 2*Lb+1/))
      !end if
  end subroutine int_schwarz
  
  {% for func in funcs %}
    {{ func }}
    
  {% endfor %}
end module mod_pa_{{ integral_name }}

module mod_pa_{{ integral_name }}

  use mod_pa_constants, only: dp, i4, PI, PI_4
  use mod_pa_boys, only: boys
  
  implicit none

  real(dp), parameter :: epsilon = 1d-10
  real(dp), parameter :: epsilon2 = epsilon**2

  type fp
      procedure({{ integral_name }}_proc), pointer, nopass :: f => null()
  end type fp

  interface
    subroutine {{ integral_name }}_proc(axs, das, A, bxs, dbs, B, cxs, dcs, C, res)
      import :: dp

      ! Orbital exponents
      real(dp), intent(in) :: axs(:), bxs(:), cxs(:)
      ! Contraction coefficients
      real(dp), intent(in) :: das(:), dbs(:), dcs(:)
      ! Centers
      real(dp), intent(in) :: A(3), B(3), C(3)
      real(dp), intent(out) :: res(:)
    end subroutine {{ integral_name }}_proc

    {% for L_tot in L_tots %}
    module subroutine {{ integral_name }}_{{ L_tot|join("") }} (axs, das, A, bxs, dbs, B, cxs, dcs, C, res)
      real(dp), intent(in) :: axs(:), bxs(:), cxs(:)
      real(dp), intent(in) :: das(:), dbs(:), dcs(:)
      real(dp), intent(in) :: A(3), B(3), C(3)
      real(dp), intent(out) :: res(:)
    end subroutine {{ integral_name }}_{{ L_tot|join("") }}
    {% endfor %}
  end interface

   type(fp) :: func_array(0:{{ lmax }}, 0:{{ lmax }}, 0:{{ lauxmax }})

contains
  subroutine {{ integral_name }}_init()
    ! Initializer procedure. MUST be called before {{ integral_name }} can be called.
    {% for L_tot in L_tots %}
        func_array({{ L_tot|join(", ") }})%f => {{ integral_name }}_{{ L_tot|join("") }}
    {% endfor %}
  end subroutine {{ integral_name }}_init

  subroutine resort_bac_abc(res, sizea, sizeb, sizec)
      real(dp), intent(in out) :: res(:)
      integer(i4), intent(in) :: sizea, sizeb, sizec
      integer(i4) :: a, b, c, i, j
      real(dp) :: tmp(size(res))

      i = 1
      do b = 1, sizeb
         do a = 1, sizea
            do c = 1, sizec
               j = sizeb*sizec*(a - 1) + sizec*(b - 1) + c
               tmp(j) = res(i)
               i = i + 1
            end do
         end do
      end do

      res = tmp
   end subroutine resort_bac_abc

  subroutine {{ integral_name }}(La, Lb, Lc, axs, das, A, bxs, dbs, B, cxs, dcs, C, res)
      integer(i4), intent(in) :: La, Lb, Lc
      ! Orbital exponents
      real(dp), intent(in) :: axs(:), bxs(:), cxs(:)
      ! Contraction coefficients
      real(dp), intent(in) :: das(:), dbs(:), dcs(:)
      ! Centers
      real(dp), intent(in) :: A(3), B(3), C(3)
      real(dp), intent(out) :: res(:)

      ! Initializing with => null () adds an implicit save, which will mess
      ! everything up when running with OpenMP.
      procedure({{ integral_name }}_proc), pointer :: fncpntr

      fncpntr => func_array(La, Lb, Lc)%f

      ! Call actual integral function
      call fncpntr(axs, das, A, bxs, dbs, B, cxs, dcs, C, res)
      
      ! Integrals in the 1d array ress are in C-order with Lc changing the fastest
      ! (loop order La, Lb, Lc).
      if (La < Lb) then
          call resort_bac_abc(res, 2*La+1, 2*Lb+1, 2*Lc+1)
      end if
  end subroutine {{ key }}
end module mod_pa_{{ integral_name }}

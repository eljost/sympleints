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

   subroutine reorder_baba_abab(res, sizea, sizeb)
      real(dp), intent(in out) :: res(:)
      integer(i4), intent(in) :: sizea, sizeb
      integer(i4) :: a, b, c, d, i, j
      real(dp) :: tmp(size(res))

      i = 1
      do b = 1, sizeb
         do a = 1, sizea
           do d = 1, sizeb
              do c = 1, sizea
                 j = sizeb * sizea * sizeb * (a - 1) + sizea * sizeb * (b - 1) + sizeb * (c-1) + d
                 tmp(j) = res(i)
                 i = i + 1
              end do
           end do
         end do
      end do
      res = tmp
   end subroutine reorder_baba_abab

  ! The procedure below is disabled for now, as this module is currently only intended to be used
  ! for the calculation of the integrals required for screening.
  !
  ! subroutine {{ integral_name }}(La, Lb, Lc, Ld, axs, das, A, bxs, dbs, B, cxs, dcs, C, dxs, dds, D, res)
  !    integer(i4), intent(in) :: La, Lb, Lc, Ld
  !    ! Orbital exponents
  !    real(dp), intent(in) :: axs(:), bxs(:), cxs(:), dxs(:)
  !    ! Contraction coefficients
  !    real(dp), intent(in) :: das(:), dbs(:), dcs(:), dds(:)
  !    ! Centers
  !    real(dp), intent(in) :: A(3), B(3), C(3), D(3)
  !    real(dp), intent(out) :: res(:)
  !
  !    ! Initializing with => null () adds an implicit save, which will mess
  !    ! everything up when running with OpenMP.
  !    procedure({{ integral_name }}_proc), pointer :: fncpntr
  !
  !    fncpntr => func_array(La, Lb, Lc, Ld)%f
  !
  !    call fncpntr(axs, das, A, bxs, dbs, B, cxs, dcs, C, dxs, dds, D, res)
  !    TODO: handle reordering and generate all functions before this can be enabled!
  ! end subroutine {{ integral_name }}

  subroutine int_schwarz (La, Lb, axs, das, A, bxs, dbs, B, res)
      integer(i4), intent(in) :: La, Lb
      ! Orbital exponents
      real(dp), intent(in) :: axs(:), bxs(:)
      ! Contraction coefficients
      real(dp), intent(in) :: das(:), dbs(:)
      ! Centers
      real(dp), intent(in) :: A(3), B(3)
      real(dp), intent(out) :: res(:)
      integer(i4) :: sizea, sizeb
      ! Initializing with => null () adds an implicit save, which will mess
      ! everything up when running with OpenMP.
      procedure({{ integral_name }}_proc), pointer :: fncpntr

      fncpntr => func_array(La, Lb, La, Lb)%f

      ! ERIs (ab|cd) are defined using chemist's notation (11|22). We are interested
      ! in the integrals <aa|bb> in physicist's notation or (ab|ab) in chemist's notation.
      call fncpntr(axs, das, A, bxs, dbs, B, axs, das, A, bxs, dbs, B, res)
      
      ! WARNING: The integrals aren't reordered from (ba|ba) to (ab|ab) when Lb > La!
      ! Maybe this subroutine should already return only the norm?
      !
      ! Reordering the integrals is not strictly needed, as we are only interested in the norm
      ! of the whole integral batch for screening purposes
      !
      ! Reorder from (ba|ba) to (abab)
      ! if (La < Lb) then
      !   sizea = 2 * La + 1
      !   sizeb = 2 * Lb + 1
      !   call reorder_baba_abab(res, sizea, sizeb)
      ! end if
  end subroutine int_schwarz
  
  {% for func in funcs %}
    {{ func }}
    
  {% endfor %}
end module mod_pa_{{ integral_name }}

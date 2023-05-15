subroutine {{ name }} (axs, das, A, bxs, dbs, B, cxs, dcs, C, dxs, dds, D, R, res)
  real(kind=real64), intent(in) :: A(3), B(3), C(3), D(3)
  real(kind=real64), intent(in) :: R(3)
  real(kind=real64), intent(in) :: axs(:), bxs(:), cxs(:), dxs(:)
  real(kind=real64), intent(in) :: das(:), dbs(:), dcs(:), dds(:)
  real(kind=real64), intent(out) :: res(:)
  
  ! Note the swapped argument order BADC instead of ABCD
  ! Currently, this is only suitable for Schwarz-type integrals (ij|ij)/<ii|jj>.
  call {{ act_name }}(bxs, dbs, B, axs, das, A, dxs, dds, D, cxs, dcs, C, R, res)
end subroutine {{ name }}

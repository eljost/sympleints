subroutine {{ name }} (axs, das, A, bxs, dbs, B, cxs, dcs, C, dxs, dds, D, res)
  real(dp), intent(in) :: A(3), B(3), C(3), D(3)
  real(dp), intent(in) :: axs(:), bxs(:), cxs(:), dxs(:)
  real(dp), intent(in) :: das(:), dbs(:), dcs(:), dds(:)
  real(dp), intent(out) :: res(:)
  
  ! Note the swapped argument order BADC instead of ABCD
  ! Currently, this is only suitable for Schwarz-type integrals (ij|ij)/<ii|jj>.
  call {{ act_name }}(bxs, dbs, B, axs, das, A, dxs, dds, D, cxs, dcs, C, res)
end subroutine {{ name }}

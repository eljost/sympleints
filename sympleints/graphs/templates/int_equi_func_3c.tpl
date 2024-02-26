subroutine {{ name }} (axs, das, A, bxs, dbs, B, cxs, dcs, C, res)
  real(dp), intent(in) :: A(3), B(3), C(3)
  real(dp), intent(in) :: axs(:), bxs(:), cxs(:)
  real(dp), intent(in) :: das(:), dbs(:), dcs(:)
  real(dp), intent(out) :: res(:)
  
  ! Note the swapped argument order BAC instead of ABC
  call {{ act_name }}(bxs, dbs, B, axs, das, A, cxs, dcs, C, res)
end subroutine {{ name }}

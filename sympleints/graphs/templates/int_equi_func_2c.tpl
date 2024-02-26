subroutine {{ name }} (axs, das, A, bxs, dbs, B, res)
  real(dp), intent(in) :: A(3), B(3)
  real(dp), intent(in) :: axs(:), bxs(:)
  real(dp), intent(in) :: das(:), dbs(:)
  real(dp), intent(out) :: res(:)
  
  ! Note the swapped argument order BA instead of AB
  call {{ act_name }}(bxs, dbs, B, axs, das, A, res)
end subroutine {{ name }}

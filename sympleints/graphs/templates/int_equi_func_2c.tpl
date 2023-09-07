subroutine {{ name }} (axs, das, A, bxs, dbs, B, R, res)
  real(kind=real64), intent(in) :: A(3), B(3)
  real(kind=real64), intent(in) :: R(3)
  real(kind=real64), intent(in) :: axs(:), bxs(:)
  real(kind=real64), intent(in) :: das(:), dbs(:)
  real(kind=real64), intent(out) :: res(:)
  
  ! Note the swapped argument order BA instead of AB
  call {{ act_name }}(bxs, dbs, B, axs, das, A, R, res)
end subroutine {{ name }}

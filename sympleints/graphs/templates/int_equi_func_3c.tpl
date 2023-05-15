subroutine {{ name }} (axs, das, A, bxs, dbs, B, cxs, dcs, C, R, res)
  real(kind=real64), intent(in) :: A(3), B(3), C(3)
  real(kind=real64), intent(in) :: R(3)
  real(kind=real64), intent(in) :: axs(:), bxs(:), cxs(:)
  real(kind=real64), intent(in) :: das(:), dbs(:), dcs(:)
  real(kind=real64), intent(out) :: res(:)
  
  ! Note the swapped argument order BAC instead of ABC
  call {{ act_name }}(bxs, dbs, B, axs, das, A, cxs, dcs, C, R, res)
end subroutine {{ name }}

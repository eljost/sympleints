subroutine {{ name }} (axs, das, A, bxs, dbs, B, cxs, dcs, C, dxs, dds, D, res)
  real(dp), intent(in) :: A(3), B(3), C(3), D(3)
  real(dp), intent(in) :: axs(:), bxs(:), cxs(:), dxs(:)
  real(dp), intent(in) :: das(:), dbs(:), dcs(:), dds(:)
  ! Orbital exponents
  real(dp) :: ax, bx, cx, dx
  ! Quantities dependent on centers A and B
  real(dp) :: px, mu, AB(3), RAB2, KAB, P(3), PA(3)
  ! Quantities dependent on centers C and D
  real(dp) :: qx, nu, CD(3), RCD2, KCD, Q(3), QC(3)
  real(dp) :: PQ(3), RPQ2, boys_arg
  ! Counters
  integer(i4) :: i, j, k, l

  ! 1D intermediate arrays
  {% for arr_name, arr in array_defs.items() %}
  {# real(dp){% if arr_name == 'res' %}, intent(out){% endif %} :: {{ arr_name }}({{ arr.shape[0] }})#}
  {% if arr_name == target_array_name %}
  ! Target array
  {% endif %}
  real(dp) :: {{ arr_name }}({{ arr.shape[0] }})
  {% endfor %}

  ! Arrays for partially contracted integrals
  real(dp) :: d_buffer({{ shell_size }})
  real(dp) :: c_buffer({{ shell_size }})
  real(dp) :: b_buffer({{ shell_size }})
  ! Final contracted integrals
  !real(dp), intent(out) :: res({{ shell_size }})
  real(dp), intent(out) :: res(:)

  AB = A - B
  RAB2 = sum(AB**2)
  CD = C - D
  RCD2 = sum(CD**2)

  ! Initialize result array
  res = 0.0d0
  ! Loop over primitives on center A
  do i = 1, size(axs)
    ax = axs(i)
    ! Loop over primitives on center B
    b_buffer = 0.0d0
    do j = 1, size(bxs)
      bx = bxs(j)
      px = ax + bx
      mu = ax * bx / px
      KAB = exp(-mu * RAB2)
      P = (ax * A + bx * B) / px
      PA = P - A
      
      c_buffer = 0.0d0
      ! Loop over primitives on center C
      do k = 1, size(cxs)
        cx = cxs(k)
        ! Loop over primitives on center D
        d_buffer = 0.0d0
        do l = 1, size(dxs)
          dx = dxs(l)
          qx = cx + dx
          nu = cx * dx / qx
          KCD = exp(-nu * RCD2)
          Q = (cx * C + dx * D) / qx
          QC = Q - C
          
          PQ = P - Q
          RPQ2 = sum(PQ**2)
          boys_arg = px * qx / (px + qx) * RPQ2
          {% for name, lines in blocks.items() %}
          ! {{ name }}, {{ lines|length }} expression(s)
          {% for line in lines %}
          {{ line }}
          {% endfor %}
          {% endfor %}
        ! Contract primitive integrals at center D
        d_buffer = d_buffer + dds(l) * {{ target_array_name }}
        ! End of loop over D
        end do
        ! Contract primitive integrals at center C
        c_buffer = c_buffer + dcs(k) * d_buffer
      ! End of loop over C
      end do
      ! Contract primitive integrals at center B
      b_buffer = b_buffer + dbs(j) * c_buffer
    ! End of loop over B
    end do
    res = res + das(i) * b_buffer
  ! End of loop over A
   end do
end subroutine {{ name }}

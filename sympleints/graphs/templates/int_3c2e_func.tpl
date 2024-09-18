module procedure {{ name }}
  ! Orbital exponents
  real(dp) :: ax, bx, cx
  ! Quantities dependent on centers A and B
  real(dp) :: px, kappa, mu, AB(3), P(3), PA(3)
  ! Quantities dependent on center C
  real(dp) :: PC(3), alpha, theta, R2PC
  ! Quantities for primitive prescreening
  real(dp) :: cx_min, theta_min2, alpha_min, prim_est
  ! Counters
  integer :: i, j, k

  ! 1D intermediate arrays
  {% for arr_name, arr in array_defs.items() %}
  {# real(dp){% if arr_name == 'res' %}, intent(out){% endif %} :: {{ arr_name }}({{ arr.shape[0] }})#}
  {% if arr_name == target_array_name %}
  ! Target array
  {% endif %}
  real(dp) :: {{ arr_name }}({{ arr.shape[0] }})
  {% endfor %}

  ! Arrays for partially contracted integrals
  real(dp) :: c_buffer({{ shell_size }})
  real(dp) :: b_buffer({{ shell_size }})

  AB = A - B
  ! Smallest orbtial exponent at C
  cx_min = minval(cxs)

  ! Loop over primitives on center A
  res = 0.0d0
  do i = 1, size(axs)
    ax = axs(i)
    ! Loop over primitives on center B
    b_buffer = 0.0d0
    do j = 1, size(bxs)
      bx = bxs(j)
      px = ax + bx
      mu = ax * bx / px
      kappa = exp(-mu * sum(AB**2))
      P = (ax * A + bx * B) / px
      PA = P - A
      PC = P - C
      R2PC = sum(PC**2)

      ! pPRE2 primitive screening
      ! Directly calculate square of theta_pc, to avoid 'sqrt' call
      theta_min2 = (2d0 * pi**(5d0/2d0) / (px * cx_min))**2 / (px + cx_min)
      alpha_min = px * cx_min / (px + cx_min)
      prim_est = theta_min2 * kappa**2 * min(1d0, PI_4 / (alpha_min * R2PC))
      ! Compare against square of 1d-10
      if (prim_est <= epsilon2) then
        continue
      end if 

      c_buffer = 0.0d0
      ! Loop over primitives on center C
      do k = 1, size(cxs)
        cx = cxs(k)
        alpha = px * cx / (px + cx)
        theta = 2d0 * pi**(5d0/2d0) / (px * cx * sqrt(px + cx))

        {% for name, lines in blocks.items() %}
          ! {{ name }}, {{ lines|length }} expression(s)
          {% for line in lines %}
            {{ line }}
          {% endfor %}
        {% endfor %}

        ! Contract primitive integrals at center C
        c_buffer = c_buffer + dcs(k) * {{ target_array_name }} 
      ! End of loop over C
      end do
      ! Contract primitive integrals at center B
      b_buffer = b_buffer + dbs(j) * c_buffer
    ! End of loop over B
    end do
    res = res + das(i) * b_buffer
  ! End of loop over A
   end do
end procedure {{ name }}

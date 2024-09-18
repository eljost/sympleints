module procedure {{ name }}
  ! Orbital exponents
  real(dp) :: ax, bx
  ! Quantities dependent on centers A and B
  real(dp) :: px, mu, AB(3), P(3), PA(3), PB(3), R2AB
  ! Counters
  integer :: i, j

  ! 1D intermediate arrays
  {% for arr_name, arr in array_defs.items() %}
  {# real(dp){% if arr_name == 'res' %}, intent(out){% endif %} :: {{ arr_name }}({{ arr.shape[0] }})#}
  {% if arr_name == target_array_name %}
  ! Target array
  {% endif %}
  real(dp) :: {{ arr_name }}({{ arr.shape[0] }})
  {% endfor %}

  ! Arrays for partially contracted integrals
  real(dp) :: b_buffer({{ shell_size }})

  AB = A - B
  R2AB = sum(AB**2)

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
      P = (ax * A + bx * B) / px
      PA = P - A
      PB = P - B

      {% for name, lines in blocks.items() %}
        ! {{ name }}, {{ lines|length }} expression(s)
        {% for line in lines %}
          {{ line }}
        {% endfor %}
      {% endfor %}

      ! Contract primitive integrals at center B
      b_buffer = b_buffer + dbs(j) * {{ target_array_name }}
    ! End of loop over B
    end do
    res = res + das(i) * b_buffer
  ! End of loop over A
   end do
end procedure {{ name }}

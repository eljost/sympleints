
{% for exp_ in exps %}
real({{ kind }}), intent(in) :: {{ exp_ }}{% if contracted %}(:){% endif %}  ! Primitive exponent(s)
{% endfor %}
{% if contracted %}
! Contraction coefficient(s)
{% for exp_, coeff in zip(exps, coeffs) %}
real({{ kind }}), intent(in), dimension(size({{ exp_ }})) :: {{ coeff }}
{% endfor %}
{% else %}
real({{ kind }}), intent(in) :: {{ coeffs|join(", ") }}
{% endif %}
! Center(s)
real({{ kind }}), intent(in), dimension(3) :: {{ centers|join(", ") }}
{% if ref_center %}
! Reference center; used only by some procedures
real({{ kind }}), intent(in), dimension(3) :: {{ ref_center }}
{% endif %}
! Return value
real({{ kind }}), intent(in out) :: {{ res_name }}(:)

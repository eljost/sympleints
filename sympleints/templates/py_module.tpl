"""
{{ header }}
"""

{% if comment %}
"""
{{ comment }}
"""
{% endif %}

import numpy

{% if boys %}
# The default boys-import can be changed via the --boys-func argument.
# sympleints also includes a basic implementation in 'sympleints.boys'
from {{ boys }} import boys
{% endif %}

{% for ai in add_imports  %}
{{ ai }}
{% endfor %}

{% for func in funcs %}
{{ func.text }}
{% endfor %}

{{ func_dict }}

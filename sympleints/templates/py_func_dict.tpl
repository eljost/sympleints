{{ name }} = {
{% for func in rendered_funcs %}
{{ func.Ls }}: {{ func.name }},
{% endfor %}
}

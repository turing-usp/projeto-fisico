{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
.. currentmodule:: {{ fullname }}

{% if modules %}
.. rubric:: Módulos
.. autosummary::
   :toctree: .
   {% for module in modules %}
   {{ module }}
   {% endfor %}
{% endif %}

{% if attributes %}
.. rubric:: Atributos
.. autosummary::
   :toctree: .
   {% for attr in attributes %}
   {{ attr }}
   {% endfor %}
{% endif %}

{% if classes %}
.. rubric:: Classes
.. autosummary::
   :toctree: .
   :nosignatures:
   {% for class in classes %}
   {{ class }}
   {% endfor %}
{% endif %}

{% if functions %}
.. rubric:: Funções
.. autosummary::
   :toctree: .
   {% for function in functions %}
   {{ function }}
   {% endfor %}
{% endif %}

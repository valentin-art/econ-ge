{#
    silver/staging/marts/dbt_artifacts/experiments already exist as fixed schemas
    (docker/postgres/init/01-init.sql). Use the model's custom schema literally
    instead of dbt's default "<target_schema>_<custom_schema>" so a single dev
    target still writes into those exact schemas.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}

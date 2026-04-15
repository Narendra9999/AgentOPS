#!/bin/bash
# Air-gapped install — all packages from UC Volume with --no-deps
VOL=/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/python312_all_libs

pip install --no-deps \
  $VOL/databricks_langchain-0.5.1-py3-none-any.whl \
  $VOL/databricks_ai_bridge-0.5.1-py3-none-any.whl \
  $VOL/databricks_sdk-0.73.0-py3-none-any.whl \
  $VOL/mlflow-3.8.1-py3-none-any.whl \
  $VOL/mlflow_skinny-3.3.2-py3-none-any.whl \
  $VOL/mlflow_tracing-3.3.2-py3-none-any.whl \
  $VOL/fastapi-0.115.12-py3-none-any.whl \
  $VOL/uvicorn-0.34.1-py3-none-any.whl \
  $VOL/requests-2.32.5-py3-none-any.whl \
  $VOL/starlette-0.46.2-py3-none-any.whl \
  $VOL/pydantic-2.10.1-py3-none-any.whl \
  $VOL/pydantic_settings-2.8.1-py3-none-any.whl \
  $VOL/click-8.1.7-py3-none-any.whl \
  $VOL/h11-0.16.0-py3-none-any.whl \
  $VOL/anyio-4.9.0-py3-none-any.whl \
  $VOL/sniffio-1.3.1-py3-none-any.whl \
  $VOL/httpx-0.28.1-py3-none-any.whl \
  $VOL/httpx_sse-0.4.3-py3-none-any.whl \
  $VOL/httpcore-1.0.9-py3-none-any.whl \
  $VOL/langchain_core-0.3.63-py3-none-any.whl \
  $VOL/langchain-0.3.21-py3-none-any.whl \
  $VOL/langchain_community-0.4.1-py3-none-any.whl \
  $VOL/langchain_text_splitters-0.3.8-py3-none-any.whl \
  $VOL/langsmith-0.1.147-py3-none-any.whl \
  $VOL/tabulate-0.9.0-py3-none-any.whl \
  $VOL/alembic-1.17.2-py3-none-any.whl \
  $VOL/flask-3.1.2-py3-none-any.whl \
  $VOL/gunicorn-23.0.0-py3-none-any.whl \
  $VOL/jinja2-3.1.4-py3-none-any.whl \
  $VOL/mako-1.3.10-py3-none-any.whl \
  $VOL/sqlparse-0.5.3-py3-none-any.whl \
  $VOL/cloudpickle-3.1.2-py3-none-any.whl \
  $VOL/entrypoints-0.4-py3-none-any.whl \
  $VOL/opentelemetry_api-1.38.0-py3-none-any.whl \
  $VOL/opentelemetry_sdk-1.38.0-py3-none-any.whl \
  $VOL/opentelemetry_semantic_conventions-0.59b0-py3-none-any.whl \
  $VOL/typing_extensions-4.15.0-py3-none-any.whl \
  $VOL/packaging-25.0-py3-none-any.whl \
  $VOL/certifi-2025.6.15-py3-none-any.whl \
  $VOL/idna-3.11-py3-none-any.whl \
  $VOL/urllib3-2.5.0-py3-none-any.whl \
  $VOL/tenacity-9.1.2-py3-none-any.whl \
  $VOL/jsonpatch-1.33-py2.py3-none-any.whl \
  $VOL/jsonpointer-3.0.0-py2.py3-none-any.whl \
  $VOL/python_dotenv-1.2.1-py3-none-any.whl \
  $VOL/typing_inspection-0.4.2-py3-none-any.whl \
  $VOL/annotated_types-0.7.0-py3-none-any.whl \
  $VOL/python_dateutil-2.9.0.post0-py2.py3-none-any.whl \
  $VOL/six-1.17.0-py2.py3-none-any.whl \
  $VOL/importlib_metadata-8.7.0-py3-none-any.whl \
  $VOL/zipp-3.23.0-py3-none-any.whl \
  $VOL/colorama-0.4.6-py2.py3-none-any.whl \
  $VOL/blinker-1.9.0-py3-none-any.whl \
  $VOL/itsdangerous-2.2.0-py3-none-any.whl \
  $VOL/werkzeug-3.1.3-py3-none-any.whl \
  $VOL/marshmallow-3.26.1-py3-none-any.whl \
  $VOL/dataclasses_json-0.6.7-py3-none-any.whl \
  $VOL/mypy_extensions-1.1.0-py3-none-any.whl \
  $VOL/requests_toolbelt-1.0.0-py2.py3-none-any.whl \
  $VOL/docker-7.1.0-py3-none-any.whl \
  $VOL/querystring_parser-1.2.4-py2.py3-none-any.whl \
  $VOL/google_auth-2.43.0-py2.py3-none-any.whl \
  $VOL/cachetools-6.2.2-py3-none-any.whl \
  $VOL/pyasn1-0.6.1-py3-none-any.whl \
  $VOL/pyasn1_modules-0.4.2-py3-none-any.whl \
  $VOL/rsa-4.9.1-py3-none-any.whl \
  $VOL/oauthlib-3.2.2-py3-none-any.whl \
  $VOL/Deprecated-1.2.15-py2.py3-none-any.whl \
  $VOL/platformdirs-4.3.7-py3-none-any.whl \
  $VOL/filelock-3.20.0-py3-none-any.whl \
  $VOL/tqdm-4.67.1-py3-none-any.whl \
  $VOL/attrs-25.4.0-py3-none-any.whl \
  $VOL/jsonschema-4.25.1-py3-none-any.whl \
  $VOL/jsonschema_specifications-2025.9.1-py3-none-any.whl \
  $VOL/referencing-0.37.0-py3-none-any.whl \
  $VOL/pyrsistent-0.19.3-py3-none-any.whl \
  $VOL/gitpython-3.1.45-py3-none-any.whl \
  $VOL/gitdb-4.0.12-py3-none-any.whl \
  $VOL/smmap-5.0.2-py3-none-any.whl

exec python server.py

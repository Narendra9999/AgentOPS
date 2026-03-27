"""
Test configuration — mock mlflow and databricks.sdk for local testing.
These packages are only available on Databricks clusters.
"""

import sys
from unittest.mock import MagicMock

# Mock mlflow before any imports
mlflow_mock = MagicMock()
mlflow_mock.trace = lambda **kwargs: lambda fn: fn  # @mlflow.trace decorator becomes no-op
mlflow_mock.update_current_trace = MagicMock()
sys.modules["mlflow"] = mlflow_mock
sys.modules["mlflow.pyfunc"] = MagicMock()
sys.modules["mlflow.types"] = MagicMock()
sys.modules["mlflow.types.agent"] = MagicMock()
sys.modules["mlflow.models"] = MagicMock()
sys.modules["mlflow.genai"] = MagicMock()
sys.modules["mlflow.genai.scorers"] = MagicMock()

# Mock databricks SDK
sys.modules["databricks"] = MagicMock()
sys.modules["databricks.sdk"] = MagicMock()
sys.modules["databricks.sdk.service"] = MagicMock()
sys.modules["databricks.sdk.service.vectorsearch"] = MagicMock()
sys.modules["databricks.sdk.service.serving"] = MagicMock()
sys.modules["databricks.sdk.service.catalog"] = MagicMock()
sys.modules["databricks.vectorsearch"] = MagicMock()
sys.modules["databricks.vectorsearch.client"] = MagicMock()
sys.modules["databricks.connect"] = MagicMock()

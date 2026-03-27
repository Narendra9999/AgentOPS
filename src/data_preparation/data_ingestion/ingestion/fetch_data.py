"""
Data Ingestion — Fetch Databricks documentation from the sitemap.
Scrapes HTML pages, extracts article text, returns a Spark DataFrame.

Data source: https://docs.databricks.com/en/doc-sitemap.xml
Reference: https://github.com/ryuta-yoshimatsu/agentops-demo
"""

from pyspark.sql.types import StringType
from pyspark.sql.functions import pandas_udf

from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Retry config for HTTP requests
retries = Retry(
    total=3,
    backoff_factor=3,
    status_forcelist=[429, 500, 502, 503, 504],
)


def fetch_sitemap_urls(sitemap_url: str, max_documents: int = None) -> list[str]:
    """Fetch all page URLs from a sitemap XML."""
    response = requests.get(sitemap_url, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    urls = [loc.text for loc in root.findall(f".//{namespace}loc")]

    logger.info(f"Found {len(urls)} URLs in sitemap")
    if max_documents:
        urls = urls[:max_documents]
        logger.info(f"Limited to {max_documents} documents")

    return urls


def load_data_from_file(spark, file_path: str, max_documents: int = None):
    """
    Load pre-downloaded documentation from a local JSON file.
    For air-gapped environments where the cluster can't reach the internet.

    Args:
        spark: SparkSession
        file_path: Path to JSON file with [{"url": "...", "text": "..."}, ...]
        max_documents: Optional limit

    Returns:
        DataFrame with columns: url, text
    """
    import json

    with open(file_path, "r") as f:
        docs = json.load(f)

    if max_documents:
        docs = docs[:max_documents]

    logger.info(f"Loaded {len(docs)} documents from {file_path}")

    df = spark.createDataFrame(docs)
    df = df.select("url", "text").filter("text IS NOT NULL")

    row_count = df.count()
    logger.info(f"Loaded {row_count} valid documents")

    if row_count == 0:
        raise Exception(f"No documents found in {file_path}")

    return df


def fetch_data_from_url(spark, data_source_url: str, max_documents: int = None):
    """
    Fetch Databricks documentation pages and extract text content.

    Args:
        spark: SparkSession
        data_source_url: URL to the sitemap XML
        max_documents: Optional limit on number of docs to fetch

    Returns:
        DataFrame with columns: url, text
    """
    # Step 1: Get all page URLs from sitemap
    urls = fetch_sitemap_urls(data_source_url, max_documents)

    # Step 2: Create Spark DataFrame of URLs
    df_urls = spark.createDataFrame(urls, StringType()).toDF("url").repartition(10)

    # Step 3: Fetch HTML content in parallel using Pandas UDF
    @pandas_udf("string")
    def fetch_html_udf(urls: pd.Series) -> pd.Series:
        adapter = HTTPAdapter(max_retries=retries)
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        def fetch_one(url):
            try:
                resp = session.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp.content
            except requests.RequestException:
                return None
            return None

        with ThreadPoolExecutor(max_workers=200) as executor:
            results = list(executor.map(fetch_one, urls))
        return pd.Series(results)

    # Step 4: Extract article text from HTML using BeautifulSoup
    @pandas_udf("string")
    def extract_text_udf(html_contents: pd.Series) -> pd.Series:
        def extract(html):
            if html:
                soup = BeautifulSoup(html, "html.parser")
                # Databricks docs use this div for article content
                article = soup.find("div", class_="theme-doc-markdown markdown")
                if article:
                    return str(article).strip()
            return None

        return html_contents.apply(extract)

    # Step 5: Apply UDFs and filter
    df_with_html = df_urls.withColumn("html_content", fetch_html_udf("url"))
    df_final = (
        df_with_html
        .withColumn("text", extract_text_udf("html_content"))
        .select("url", "text")
        .filter("text IS NOT NULL")
    )

    row_count = df_final.count()
    logger.info(f"Successfully fetched {row_count} documents")

    if row_count == 0:
        raise Exception(
            "No documents fetched. The HTML structure may have changed. "
            "Check the div class in extract_text_udf."
        )

    return df_final

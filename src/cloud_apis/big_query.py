from google.cloud import bigquery
import re
import apache_beam as beam
from apache_beam.utils.options import PipelineOptions

def query_github():
    client = bigquery.Client()
    query_results = client.run_sync_query("""
        SELECT
            APPROX_TOP_COUNT(corpus, 10) as title,
            COUNT(*) as unique_words
        FROM `publicdata.samples.shakespeare`;""")

    # Use standard SQL syntax for queries.
    # See: https://cloud.google.com/bigquery/sql-reference/
    query_results.use_legacy_sql = False

    query_results.run()

    # Drain the query results by requesting a page at a time.
    page_token = None

    while True:
        rows, total_rows, page_token = query_results.fetch_data(
            max_results=10,
            page_token=page_token)

        for row in rows:
            print(row)

        if not page_token:
            break



p = beam.Pipeline(options=PipelineOptions())
weather_data = p | beam.io.Read(
    'ReadYearAndTemp',
    beam.io.BigQuerySource(
        query='SELECT year, mean_temp FROM `samples.weather_stations`',
        use_standard_sql=True))



if __name__ == '__main__':
    query_shakespeare()

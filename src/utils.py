from google.cloud import storage
import os
from io import StringIO
import pandas as pd

def write_csv(pandas_df, bucket_name, file_name, index=True):
    # writes csv to google cloud bucket
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/opt/private_keys/recessionml_sa.json'
    f = StringIO()
    pandas_df.to_csv(f, index=index)
    f.seek(0)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    blob.upload_from_file(f, content_type='text/csv')

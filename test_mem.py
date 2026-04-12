import pandas as pd
import pyarrow.parquet as pq

try:
    print("Loading 80 columns...")
    cols = pq.ParquetFile("train.parquet").schema.names[:80]
    df = pd.read_parquet("train.parquet", columns=cols)
    print("Success 80")
    
    print("Loading 120 columns...")
    cols = pq.ParquetFile("train.parquet").schema.names[:120]
    df = pd.read_parquet("train.parquet", columns=cols)
    print("Success 120")
except Exception as e:
    import traceback
    traceback.print_exc()

import pandas as pd
import pyarrow.parquet as pq

try:
    print("Loading test cols...")
    cols = pq.ParquetFile("test.parquet").schema.names
    print(f"Total cols in test schema: {len(cols)}")
    
    # Try reading the first 100
    df = pd.read_parquet("test.parquet", columns=cols[:100])
    print("Success 100")
    
    df = pd.read_parquet("test.parquet", columns=cols[:200])
    print("Success 200")
    
    df = pd.read_parquet("test.parquet", columns=cols)
    print("Success all")

except Exception as e:
    import traceback
    traceback.print_exc()

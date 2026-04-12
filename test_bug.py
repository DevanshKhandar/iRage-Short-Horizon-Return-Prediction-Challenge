import pandas as pd

try:
    print("Loading test separately...")
    a = pd.read_parquet("test.parquet", columns=["ID"])
    b = pd.read_parquet("test.parquet", columns=["CV_GROUP"])
    print("Success loading separately!")
    
    print("Loading test together...")
    c = pd.read_parquet("test.parquet", columns=["ID", "CV_GROUP"])
    print("Success loading together!")
except Exception as e:
    import traceback
    traceback.print_exc()

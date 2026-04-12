import pyarrow.parquet as pq

train_cols = pq.ParquetFile("train.parquet").schema.names
test_cols = pq.ParquetFile("test.parquet").schema.names

print("SO3_T in train:", "SO3_T" in train_cols)
print("SO3_T in test:", "SO3_T" in test_cols)

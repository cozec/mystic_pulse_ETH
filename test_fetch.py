import sys
import os
sys.path.append('src')
from web_utils import get_eth_data_with_live
import time

print("Starting fetch test...")
start = time.time()
try:
    df = get_eth_data_with_live()
    print(f"Data fetched! Shape: {df.shape}")
    print(f"Last date: {df.index[-1]}")
    print(f"Time taken: {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error: {e}")

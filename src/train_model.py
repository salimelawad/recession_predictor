import pandas as pd
import data_ingestor

api_key = '6dcd215d39286437b67f6389b88903e0'

def generate_future_rec_counts(usrec):
    return usrec.shift(-10).iloc[::-1].rolling(6, min_periods=0).sum().iloc[::-1]

print(data_ingestor.load_recession(api_key))
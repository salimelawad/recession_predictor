import requests
import pandas as pd
import time

api_key = '6dcd215d39286437b67f6389b88903e0'

def get_series_request(series, api_key):
    query_results = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={api_key}&file_type=json").json()
    query_results = [(point['date'], point['value']) for k in ('date', 'value') for point in query_results['observations']]
    idx, values = zip(*query_results)
    df = pd.to_numeric(pd.Series(values, idx), errors='coerce')
    df = df.loc[~df.index.duplicated(keep='first')]
    df.index = pd.to_datetime(df.index)
    return df


def load_time_series_list(code_list, api_key):
    data =  [get_series_request(series, api_key) for series in code_list]
    df = pd.concat(data, axis=1, keys=code_list, join='inner')
    return df


def load_monthly(api_key):
    # Monthly N-Year Treasury Constant Maturity Minus Federal Funds Rate since Jul 1954
    historical_monthly_codes = ['TB3SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM'] 
    return load_time_series_list(historical_monthly_codes, api_key)


def load_daily(api_key):
    # Monthly N-Year Treasury Constant Maturity Minus Federal Funds Rate since Jul 1954
    monthly_codes = ['T3MFF', 'T1YFF', 'T5YFF', 'T10YFF']
    return load_time_series_list(monthly_codes, api_key)


def load_recession(api_key):
    return get_series_request('USREC', api_key)


print(load_monthly(api_key))
print(load_daily(api_key))
print(load_recession(api_key))




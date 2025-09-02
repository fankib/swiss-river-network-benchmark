import pandas
import datetime
import numpy as np

# constants

# six year dataset
UNIX_01_01_2012:int = 1325376000
UNIX_31_12_2017:int = 1514678400

# love time splits (in utc):
UNIX_01_01_1980:int =  315532800
UNIX_31_12_1989:int =  631065600
UNIX_01_01_1990:int =  631152000
UNIX_01_01_2015:int = 1420070400
UNIX_01_01_2019:int = 1546300800
UNIX_01_01_2010:int = 1262304000
UNIX_31_12_2014:int = 1419984000
UNIX_31_12_2018:int = 1546214400
UNIX_31_12_2019:int = 1577750400
UNIX_31_12_2020:int = 1609372800
UNIX_31_12_2022:int = 1672531200

def unix_timestamp(dataframe:pandas.DataFrame, format:str):
    ''' converts the given dataframe row into our unix based timestamp
    '''
    return pandas.to_numeric(pandas.to_datetime(dataframe, format=format), downcast='integer') // 1000000000

def unix_days(dataframe_with_timestamp:pandas.DataFrame):
    return dataframe_with_timestamp // 86400 # seconds per day

def from_unix_days(dataframe_with_epoch_day):
    return dataframe_with_epoch_day * 86400 # seconds per day

def to_unix_days(timestamp:int) -> int:
    return timestamp // 86400

def to_datetime(timestamp:int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(timestamp).astimezone(datetime.timezone.utc)#.replace(tzinfo=datetime.timezone.utc)

def to_human(timestamp:int) -> str:
    return str(to_datetime(timestamp))

def from_human(day, month, year) -> str:
    dt = datetime.datetime(year, month, day)
    timestamp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return int(timestamp)
import pandas as pd
import numpy as np
from openmeteo_requests import Client
from requests_cache import CachedSession
from retry_requests import retry
import joblib
import warnings
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.api import Holt

def main():
    params = readParams()
    features = convertParamsToFeatures(params)

    if not features.empty:
        predictions = predictNumberOfViewers(features)

        result = beautify(params, features, predictions)
        result.to_csv('./predictions.csv', index=False, sep=';')
    else:
        print("No valid data remaining to predict.") 

def readParams():
    params = pd.read_csv('./data.csv', delimiter=';')
    params['Datum'] = pd.to_datetime(params['Datum'], format='%d/%m/%Y', errors='coerce')
    params['Start'] = pd.to_timedelta(params['Start'], errors='coerce')
    params['Duur'] = pd.to_timedelta(params['Duur'], errors='coerce')
    params.dropna(inplace=True)
    params.rename(columns={
        'Programma': 'title',
        'Zender': 'channel',
        'Datum': 'date',
        'Start': 'startTime',
        'Duur': 'duration'
    }, inplace=True)
    return params

def convertParamsToFeatures(params):
    features = params.copy()
    features = addDescriptionFeatures(features)
    features = addChannelFeatures(features)
    features = addDateResultFeatures(features)
    features = addStartTimeFeatures(features)
    features = addrLengthFeatures(features)
    features = addHistoricalFeatures(features)
    features.dropna(inplace=True)
    return features

def addDescriptionFeatures(features):
    categories = pd.read_csv('./util/categories.csv', usecols=['title', 'category'])
    features = features.merge(categories, on='title', how='left')
    features.fillna('unknown', inplace=True)
    features['category'] = features['category'].astype('category')

    totalOccurences = pd.read_csv('./util/lastOccurrences.csv', usecols=['title', 'totalOccurrences'])
    features = features.merge(totalOccurences, on='title', how='left')
    features.fillna({'totalOccurrences': 0}, inplace=True)
    features['totalOccurrences'] = features['totalOccurrences'] + 1

    yearlyOccurences = pd.read_csv('./util/lastOccurrences.csv', usecols=['title', 'year', 'yearlyOccurrences'])
    features['year'] = features['date'].dt.year
    features = features.merge(yearlyOccurences, on=['title', 'year'], how='left')
    features.drop(columns=['year'], inplace=True)
    features.fillna({'yearlyOccurrences': 0}, inplace=True)
    features['yearlyOccurrences'] = features['yearlyOccurrences'] + 1

    return features

def addChannelFeatures(features):
    possibleChannels = pd.read_csv('./util/channels.csv', header=None).iloc[:, 0].values
    features['channel'] = features['channel'].where(features['channel'].isin(possibleChannels), 'OTHER')
    features['channel'] = features['channel'].astype('category')

    primeChannels = pd.read_csv('./util/primeChannels.csv', header=None).iloc[:, 0].values
    features['isPrimeChannel'] = features['channel'].isin(primeChannels)
    return features

def addDateResultFeatures(features):
    features['dayOfWeek'] = features['date'].dt.dayofweek
    features['dayOfWeek_sin'] = np.sin(2 * np.pi * features['date'].dt.dayofweek / 7)
    features['weekOfYear_cos'] = np.cos(2 * np.pi * features['date'].dt.isocalendar().week.astype(int) / 52)
    features['month'] = features['date'].dt.month

    weather = getWeather()
    features = features.merge(weather, on=['date'], how='left')

    return features

def getWeather():
    cache_session = CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = Client(session = retry_session)

    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
      'latitude': 50.85045,
      'longitude': 4.34878,
      'daily': ['apparent_temperature_mean'],
      'timezone': 'Europe/Berlin'
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_data = {'date': pd.date_range(
      start = pd.to_datetime(daily.Time(), unit = 's', utc = True),
      end = pd.to_datetime(daily.TimeEnd(), unit = 's', utc = True),
      freq = pd.Timedelta(seconds = daily.Interval()),
      inclusive = 'left'
    )}
    daily_data['date'] = pd.to_datetime(daily_data['date']).tz_localize(None).normalize()
    daily_data['apparentTemperature'] = daily.Variables(0).ValuesAsNumpy()
    from shutil import rmtree
    if rmtree:
        rmtree('.cache', ignore_errors=True)
    return pd.DataFrame(daily_data)

def addStartTimeFeatures(features):
    features['startTimeInSeconds'] = features['startTime'].dt.total_seconds().astype(int)
    features['timeslot'] = features['startTime'].apply(timeslot).astype('category')
    return features

def timeslot(startTime):
    if startTime.components.hours < 6:
        return 'Night'
    if startTime.components.hours < 12:
        return 'Morning'
    if startTime.components.hours < 14:
        return 'Lunch'
    if startTime.components.hours < 17:
        return 'Afternoon'
    if startTime.components.hours < 19:
        return 'Evening'
    if startTime.components.hours <= 22 and startTime.components.minutes <= 30:
        return 'PrimeTime'
    return 'LateNight'

def addrLengthFeatures(features):
    features['durationCategory'] = features['duration'].apply(durationCategory).astype('category')
    return features

def durationCategory(duration):
    if duration.components.hours < 1 and duration.components.minutes < 30:
        return 'short'
    if duration.components.hours < 1 and duration.components.minutes < 45:
        return 'medium'
    if duration.components.hours < 1:
        return 'long'
    return 'very long'

def addHistoricalFeatures(features):
    features['SMA'] = features['title'].apply(addSMA)
    features.dropna(subset=['SMA'], inplace=True)

    features['SES'] = features['title'].apply(addSES)
    features.dropna(subset=['SES'], inplace=True)

    features['DES'] = features['title'].apply(addDES)
    features.dropna(subset=['DES'], inplace=True)

    return features

def addSMA(title):
    timeSeriesData = pd.read_csv('./util/timeSeriesData.csv')
    timeSeriesData['date'] = pd.to_datetime(timeSeriesData['date']).dt.normalize()
    timeSeriesData.sort_values(by='date', inplace=True)
    timeSeriesData = timeSeriesData[timeSeriesData['title'] == title].tail(2)

    if len(timeSeriesData) == 0:
        return pd.Series(np.nan)
    if len(timeSeriesData) == 1:
        return pd.Series(timeSeriesData['numberOfViewers'].values[0])
    return pd.Series(timeSeriesData['numberOfViewers'].mean())

def addSES(title):
    timeSeriesData = pd.read_csv('./util/timeSeriesData.csv')
    timeSeriesData['date'] = pd.to_datetime(timeSeriesData['date']).dt.normalize()
    timeSeriesData.sort_values(by='date', inplace=True)
    timeSeriesData = timeSeriesData[timeSeriesData['title'] == title].set_index('date', drop=True)['numberOfViewers'].values

    if len(timeSeriesData) <= 1:
        return pd.Series(np.nan)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        model = SimpleExpSmoothing(timeSeriesData)
        fittedvalues = model.fit(
            smoothing_level=0.5,
            optimized=False
        ).fittedvalues

    lagged_forecast = np.roll(fittedvalues, 1)
    return pd.Series(lagged_forecast[-1])

def addDES(title):
    timeSeriesData = pd.read_csv('./util/timeSeriesData.csv')
    timeSeriesData['date'] = pd.to_datetime(timeSeriesData['date']).dt.normalize()
    timeSeriesData.sort_values(by='date', inplace=True)
    timeSeriesData = timeSeriesData[timeSeriesData['title'] == title].set_index('date', drop=True)['numberOfViewers'].values

    if len(timeSeriesData) <= 1:
        return pd.Series(np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        model = Holt(timeSeriesData)
        fittedvalues = model.fit(
            smoothing_level=0.5,
            smoothing_trend=0.3,
            optimized=False
        ).fittedvalues

    lagged_forecast = np.roll(fittedvalues, 1)
    return pd.Series(lagged_forecast[-1])

def predictNumberOfViewers(features):
    print('Predicting number of viewers...')

    model = joblib.load('./model.pkl')
    predictions = model.predict(features)

    print('Prediction completed.')

    return predictions

def beautify(params, features, predictions):
    params = params.loc[features.index].reset_index(drop=True)
    params['Predicted Number of Viewers'] = predictions

    params['startTime'] = params['startTime'].astype(str).str.split(' ').str[-1]
    params['duration'] = params['duration'].astype(str).str.split(' ').str[-1]

    return params

main()

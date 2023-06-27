import numpy as np 
import pandas as pd

data = np.loadtxt('sta_daily.csv')
start_date = '1981-01-01'
end_date = '2020-12-31'
daily_ticks = pd.date_range(start=start_date, end=end_date, freq='D')
daily_ticks_filtered = daily_ticks[~((daily_ticks.month == 2) & (daily_ticks.day == 29))]

df = pd.DataFrame(daily_ticks_filtered, columns=['Time'])
data_df = pd.DataFrame(data)
df2 = pd.concat([df, data_df], axis=1)
df2.to_csv('NEA_daily_rainfall_1981_2020.csv')
pause = 1
import pandas as pd
import matplotlib.pyplot as plt

#Reading data from CSV files
df = pd.read_csv('production_data.csv', index_col='Date', parse_dates=True)
#print(df)

#Handles missing data (common due to power outages)
df_filled = df.fillna(method='ffill')
#print(df_filled)


#visualize time series from production_data csv file
df.plot()
plt.show()

#visualize time series as Bar from production_data csv file
df.resample('M').sum().plot(kind='bar')
plt.show()
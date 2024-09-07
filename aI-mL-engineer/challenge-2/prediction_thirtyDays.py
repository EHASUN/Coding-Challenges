#in the production_data csv there are not sufficient fileds to calculate predictions last 30 days thats
#  why i assume necessary fields and develo below model.


import pandas as pd                           #data manipulation
import seaborn as sns                         #data visualization
import matplotlib.pyplot as plt               #data visualization

#Load the dataset
df = pd.read_csv('production_data.csv')

#Display the dataset information
df.info()

#Data type conversion and missing value handling
#Convert date field to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.info()


#Display the first few rows of the dataset
df.head().T

#Checking duplicate values
df.duplicated().sum()

#Check for missing values
print(df.isnull().sum())

#Renaming columns and correcting fields
#Sizing up the null values for wip
#Check if wip null rows correspond to 'finishing' department
null_wip = df[(df['wip'].isnull())]
null_wip

#Count the instances of unique Department entries
df['department'].value_counts()

#Check unique field entries
df['department'].unique()

#Replace the Department field entries
df['department'] = df['department'].replace({'sweing': 'sewing', 'finishing ': 'finishing'})
#Recheck if the changes are reflected
df['department'].value_counts()

#Filter rows that satisfy the criteria
df_dummy = df[(df['department'] == 'finishing') & df['wip'].isnull()]
df_dummy

#Fill NULL entries with 0
df['wip'] = df['wip'].fillna(0)
df['wip'].info()

#Check unique field entries
df['Date'].unique()

#Check unique field entries
df['quarter'].unique()

#Rename the column field
df = df.rename(columns={'quarter': 'week'})
#Rename the field entries
df['week'] = df['week'].replace({'Quarter1': 'Week1', 'Quarter2': 'Week2','Quarter3': 'Week3','Quarter4': 'Week4','Quarter5': 'Week5',})
df['week'].unique()

#Check the unique field entries
df['day'].unique()

#Check the unique entries
df['team'].unique()


#EXPLORATORY DATA ANALYSIS
#Plot histograms for all columns
df.hist(bins = 10, figsize=(15,15))
plt.tight_layout()
plt.show()

#Summary statistics for the dataset
df.describe().T


# Create boxplots for all features
plt.figure(figsize=(20, 6))
sns.boxplot(data=df)
plt.title('Boxplots for Numerical Variables prior to removing outliers')
plt.show()

#Box plots to identify outliers in numeric columns by departments
import numpy as np
numeric_cols = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(y=df[col], x=df['department'])
    plt.title(f'Box plot of {col}')
plt.tight_layout()
plt.show()

#Treatment of outliers

#Remove outliers
df_clean = df.copy()
for col in numeric_cols:
    df_clean = df_clean[df_clean[col] <= df_clean[col].quantile(0.99)]

#Create boxplots for all features
plt.figure(figsize=(20, 6))
sns.boxplot(data=df_clean)
plt.title('Boxplots for Numerical Variables after removing outliers')
plt.show()

#Number of observations before wrangling
print("Number of data before wrangling: ",len(df))

#Number of observations after wrangling
print("Number of data after wrangling: ",len(df_clean))

#Number of data removed after wrangling
print("Number of data removed: ",len(df)-len(df_clean))


#Aggregate the data
agg_df = df_clean.groupby('Date').agg({'targeted_productivity': 'mean', 'actual_productivity': 'mean'}).reset_index()

#Plot the aggregated line chart
plt.figure(figsize=(10, 6))

#Plot aggregated targeted productivity
plt.plot(agg_df['Date'], agg_df['targeted_productivity'], label='Mean Targeted Productivity', marker='o')

#Plot aggregated actual productivity
plt.plot(agg_df['Date'], agg_df['actual_productivity'], label='Mean Actual Productivity', marker='o')

#Set labels and title
plt.xlabel('Date')
plt.ylabel('Productivity - Mean')
plt.title('Aggregated Targeted vs Actual Productivity Over Time')

#Show legend
plt.legend()

#Show the plot
plt.show()

#analyze productivity variations by week of the month.
#Aggregate the data
agg_df = df_clean.groupby('week').agg({'targeted_productivity': 'mean', 'actual_productivity': 'mean'}).reset_index()

#Plot the aggregated line chart
plt.figure(figsize=(10, 6))

#Plot aggregated targeted productivity
plt.plot(agg_df['week'], agg_df['targeted_productivity'], label='Mean Targeted Productivity', marker='o')

#Plot aggregated actual productivity
plt.plot(agg_df['week'], agg_df['actual_productivity'], label='Mean Actual Productivity', marker='o')

#Set labels and title
plt.xlabel('Week')
plt.ylabel('Productivity (Mean)')
plt.title('Aggregated Targeted vs Actual Productivity Over Time')

#Show legend
plt.legend()

#Show the plot
plt.show()


#analyze productivity variations by day of the week.
#Aggregate the data
agg_day = df_clean.groupby('day').agg({'targeted_productivity': 'mean', 'actual_productivity': 'mean'}).reset_index()

#Plot the aggregated line chart
plt.figure(figsize=(10, 6))

#Plot aggregated targeted productivity
plt.plot(agg_day['day'], agg_day['targeted_productivity'], label='Mean Targeted Productivity', marker='o')

#Plot aggregated actual productivity
plt.plot(agg_day['day'], agg_day['actual_productivity'], label='Mean Actual Productivity', marker='o')

#Set labels and title
plt.xlabel('Day')
plt.ylabel('Productivity (Mean)')
plt.title('Aggregated Targeted vs Actual Productivity Over Time')

#Show legend
plt.legend()

#Show the plot
plt.show()

#Productivity by Team
#Aggregate the data
agg_df = df_clean.groupby('team').agg({'targeted_productivity': 'mean', 'actual_productivity': 'mean'}).reset_index()

#Plot the aggregated line chart
plt.figure(figsize=(10, 6))

#Plot aggregated targeted productivity
plt.plot(agg_df['team'], agg_df['targeted_productivity'], label='Mean Targeted Productivity', marker='o')

#Plot aggregated actual productivity
plt.plot(agg_df['team'], agg_df['actual_productivity'], label='Mean Actual Productivity', marker='o')

#Set labels and title
plt.xlabel('Team')
plt.ylabel('Productivity (mean)')
plt.title('Aggregated Targeted vs Actual Productivity by team')

#Show legend
plt.legend()

#Show the plot
plt.show()

#create a new feature for the difference between targeted and actual productivity and analyze the productivity by team. This is being done to better visualize the earlier analysis.
#Create new feature difference on target and actual productivity
df_clean['productivity_difference'] = df['actual_productivity'] - df['targeted_productivity']
#Aggregate the data
agg_df = df_clean.groupby('team').agg({'productivity_difference': 'mean'}).reset_index()

#Plot the aggregated line chart
plt.figure(figsize=(10, 6))

#Plot aggregated productivity difference
plt.plot(agg_df['team'], agg_df['productivity_difference'], label='Mean Productivity Difference', marker='o')

#Add a horizontal line
plt.axhline(y=0, color='red', linestyle='--', label='Zero Difference')

#Set labels and title
plt.xlabel('Team')
plt.ylabel('Productivity difference (mean)')
plt.title('Productivity by team')

#Show legend
plt.legend()

#Show the plot
plt.show()

#analyze overtime trends by team.
#Check trend on overtime by team
#Create line plot
ax = sns.lineplot(data=df_clean, x="team", y="over_time")

#Calculate the mean value of over time
mean_value = df_clean["over_time"].mean()

#Add a horizontal line at the mean value
ax.axhline(y=mean_value, color='red', linestyle='--', label='Mean Overtime')

#Set title
plt.title('Overtime Trend Between Team')

#Display legend
plt.legend()

#Show plot
plt.show()

#analyze the relationship between actual productivity and incentive.
#Check relation between actual_productivity vs incentive
sns.regplot(data=df_clean[df_clean['incentive'] > 0], x='incentive', y='actual_productivity')

#analyze the relationship between actual productivity and overtime.
#Check relation between actual_productivity vs over_time
sns.regplot(data=df_clean, x='actual_productivity', y='over_time')
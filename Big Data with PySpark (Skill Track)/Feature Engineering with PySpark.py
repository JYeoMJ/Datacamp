## Feature Engineering with PySpark

# Preliminaries -------------------------

import sys

spark.version # Check spark version
sys.version_info # Check python version

# Read a parquet file to a PySpark DataFrame
df = spark.read.parquet('example.parq')

df.count() # row count
df.columns # list columns
df.dtypes # check datatypes

## Exploratory Data Analysis:
Y_df = df.select(['LISTPRICE']) # Select dependent variable

# Display summary statistics
Y_df.describe().show()

## Other Descriptive Functions (mean, skewness etc.):

pyspark.sql.functions.mean(col)
pyspark.sql.functions.skewness(col)
pyspark.sql.functions.min(col)

cov(col1, col2) # covariance
corr(col1, col2) # correlation

# Example with mean() - Aggregate Function
# Return average(mean) of values in group
df.agg({'SALESCLOSEPRICE': 'mean'}).collect()

# Example with cov()
df.cov('SALESCLOSEPRICE','YEARBUILT')

## Plotting for PySpark DataFrames (requires conversion to Pandas):

import seaborn as sns

# CAUTION: Sample PySpark DataFrames before converting to Pandas!
# Converting large datasets can cause Pandas to crash.

# sample(withReplacement, fraction, seed)
df.sample(False, 0.5, 42).count()

# Select a single column and sample and convert to pandas
sample_df = df.select(['SALESCLOSEPRICE']).sample(False, .5, 42)
pandas_df = sample_df.toPandas()

# Plot distribution of pandas_df and display plot
sns.distplot(pandas_df)
plt.show()

## Linear Model Plot (relationship b/w variables):

# Select a the relevant columns and sample
sample_df = df.select(['SALESCLOSEPRICE', 'LIVINGAREA']).sample(False, .5, 42)

# Convert to pandas dataframe
pandas_df = sample_df.toPandas()

# Linear model plot of pandas_df
sns.lmplot(x='LIVINGAREA', y='SALESCLOSEPRICE', data=pandas_df)
plt.show()

## Dropping Data

# Inspect fields that are to be dropped
df.select(['NO','UNITNUMBER','CLASS']).show()

# Drop columns in list
cols_to_drop = ['NO','UNITNUMBER','CLASS']
df = df.drop(*cols_to_drop) 

## Text Filtering

# Filter dataframe where POTENTIALSHORTSALE field is NOT like the string 'Not Disclosed'
df = df.where(~df['POTENTIALSHORTSALE'].like('Not Disclosed'))

## Outlier Value Filtering
# Exclude values ± 3 standard deviations from mean

# Calculate values used for filtering
std_val = df.agg({'SALESCLOSEPRICE': 'stddev'}).collect()[0][0]
mean_val = df.agg({'SALESCLOSEPRICE': 'mean'}).collect()[0][0]

# Create three standard deviation (± 3) upper and lower bounds for data
hi_bound = mean_val + (3 * std_val)
low_bound = mean_val - (3 * std_val)

# Use where() to filter the DataFrame between values
df = df.where((df['LISTPRICE'] < hi_bound) & (df['LISTPRICE'] > low_bound))

## Dropping NA's or NULLs

# Drop any records with NULL values
df = df.dropna()
# Drop records if both LISTPRICE and SALESCLOSEPRICE are NULL
df = df.dropna(how='all', subset=['LISTPRICE', 'SALESCLOSEPRICE'])
# Drop records where at least two columns have NULL values
df = df.dropna(thresh=2)

## Note .dropna() parameter arguments:

# how : {‘any’, ‘all’}, default ‘any’ - Determine if row or column is removed from DataFrame, when we have at least one NA or all NA
# thresh : int - drop records that have less than thresh non-null values. This overwrites the how parameter
# subset : column label or sequence of labels - optional list of column names to consider

## Dropping Duplicates

# Entire DataFrame
df.dropDuplicates()
# Check only a column list
df.dropDuplicates(['streetaddress'])

## Adjusting Data - Minmax Scaling (transform range between 0 and 1)

# Define min and max values and collect them
max_days = df.agg({'DAYSONMARKET': 'max'}).collect()[0][0]
min_days = df.agg({'DAYSONMARKET': 'min'}).collect()[0][0]
# Create a new column based on the scaled data
df = df.withColumn("scaled_days", (df['DAYSONMARKET'] - min_days) / (max_days - min_days))
# Show the first 5 rows of the 'scaled_days' column
df[['scaled_days']].show(5)

## Adjusting Data - Standardization

# Calculate the mean and standard deviation of the 'DAYSONMARKET' column
mean_days = df.agg({'DAYSONMARKET': 'mean'}).collect()[0][0]
stddev_days = df.agg({'DAYSONMARKET': 'stddev'}).collect()[0][0]

# Create a new column with the scaled data (Z-score normalization)
df = df.withColumn("ztrans_days", (df['DAYSONMARKET'] - mean_days) / stddev_days)

# Check the mean and stddev of the 'ztrans_days' standardized column
df.agg({'ztrans_days': 'mean'}).collect()
df.agg({'ztrans_days': 'stddev'}).collect()

## Adjusting Data - Log Scaling (for working with skewed data)

# import the log function
from pyspark.sql.functions import log

# Recalculate the logarithm of SALESCLOSEPRICE and create a new column 'log_SalesClosePrice'
df = df.withColumn('log_SalesClosePrice', log(df['SALESCLOSEPRICE']))

## Working with Missing Data

# Count of missing data for variable 'ROOF'
df.where(df['ROOF'].isNull()).count()

# Plotting Missing Values

import seaborn as sns

# Subset and sample the DataFrame, convert to Pandas
sub_df = df.select(['ROOMAREA1'])
sample_df = sub_df.sample(False, 0.5, seed=4)
pandas_df = sample_df.toPandas()

# Missing Values Heatmap
sns.heatmap(data=pandas_df.isnull())

## Imputation of Missing Values

# Syntax: fillna(value, subset=None) 
# value to replace missing values with, subset for column names to apply to

# Replacing missing values with zero
df.fillna(0, subset=['DAYSONMARKET'])

# Replacing missing values with the mean value for the 'DAYSONMARKET' column
col_mean = df.agg({'DAYSONMARKET': 'mean'}).collect()[0][0]
df = df.fillna(col_mean, subset=['DAYSONMARKET'])

## PySpark DataFrame Joins

# Join syntax:

DataFrame.join(
	other,			# Other DataFrame to merge
	on = None,		# The keys to join on
	how = None)		# Type of join to perform (default is 'inner')

# Inspect dataframe head
hdf.show(2)

# Specify the join condition
cond = [df['OFFMARKETDATE'] == hdf['dt']]
# Join 'hdf' onto 'df'
df = df.join(hdf, on=cond, how='left')
# Count the number of sales that occurred on bank holidays
num_sales_on_bank_holidays = df.where(~df['nm'].isNull()).count()

## SparkSQL Join

# Register the dataframe as a temp table
df.createOrReplaceTempView("df")
hdf.createOrReplaceTempView("hdf")

# SQL Query for Join (all columns from 'df' and matching rows from 'hdf')
sql_df = spark.sql("""
	SELECT * FROM df
	LEFT JOIN hdf
	ON df.OFFMARKETDATE = hdf.dt
	""")

## Feature Generation

# Creating a new feature, area by multiplying
df = df.withColumn('TSQFT', (df['WIDTH'] * df['LENGTH']))
# Sum two columns
df = df.withColumn('TSQFT', (df['SQFTBELOWGROUND'] + df['SQFTABOVEGROUND']))
# Divide two columns
df = df.withColumn('PRICEPERTSQFT', (df['LISTPRICE'] / df['TSQFT']))
# Difference two columns
df = df.withColumn('DAYSONMARKET', datediff('OFFMARKETDATE','LISTDATE'))

## \\ Time Features \\

## Treating Date Fields as Dates

from pyspark.sql.functions import to_date

# Cast data type to Date and inspect field
df = df.withColumn('LISTDATE', to_date('LISTDATE'))
df[['LISTDATE']].show(2)

## Time Components

from pyspark.sql.functions import year, month

# Create a new column of year number
df = df.withColumn('LIST_YEAR', year('LISTDATE'))
# Create a new column of month number
df = df.withColumn('LIST_MONTH', month('LISTDATE'))

from pyspark.sql.functions import dayofmonth, weekofyear

# Create new columns of the day number within the month
df = df.withColumn('LIST_DAYOFMONTH', dayofmonth('LISTDATE'))
# Create new columns of the week number within the year
df = df.withColumn('LIST_WEEKOFYEAR', weekofyear('LISTDATE'))

## Basic Time-Based Metrics

from pyspark.sql.functions import datediff

# Calculate difference between two date fields
df.withColumn('DAYSONMARKET', datediff('OFFMARKETDATE','LISTDATE'))

## Lagging Features

from pyspark.sql.functions import lag
from pyspark.sql.window import Window

# Create Window
w = Window().orderBy(m_df['DATE'])
# Create lagged column
m_df = m_df.withColumn('MORTGAGE-1wk', lag('MORTGAGE', count=1).over(w))

# Inspect results
m_df.show(3)

## \\ Extracting Features \\

## Extract Age with Text Match

from pyspark.sql.functions import when

# Create boolean filters (% wildcard for any no. characters before or after)
find_under_8 = df['ROOF'].like('%Age 8 Years or Less%')
find_over_8 = df['ROOF'].like('%Age Over 8 Years%')

# Apply filters using when() and otherwise()
# String variable converted into Boolean variable

df = df.withColumn('old_roof', 
                  when(find_over_8, 1)
                  .when(find_under_8, 0)
                  .otherwise(None))

# Inspect results
df[['ROOF', 'old_roof']].show(3, truncate=100)

## Splitting Columns (for Roof Material)

from pyspark.sql.functions import split

# Split the column on commas into a list
split_col = split(df['ROOF'], ',') # split on delimiter

# Put the first value of the list into a new column 'Roof_Material'
df = df.withColumn('Roof_Material', split_col.getItem(0))

# Inspect the results
df[['ROOF', 'Roof_Material']].show(5, truncate=100)

## Explode & Pivot

from pyspark.sql.functions import split, explode, lit, coalesce, first

# Split the column on commas into a list
df = df.withColumn('roof_list', split(df['ROOF'], ','))

# Explode the list into new records for each value
ex_df = df.withColumn('ex_roof_list', explode(df['roof_list']))

# Create a dummy column of constant value
ex_df = ex_df.withColumn('constant_val', lit(1))

# Pivot the values into boolean columns
piv_df = ex_df.groupBy('NO').pivot('ex_roof_list').agg(coalesce(first('constant_val')))

## \\ Binarizing, Bucketing and Encoding \\

## Binarizing (Convert to 1-0)

from pyspark.ml.feature import Binarizer
# Cast the data type to double
df = df.withColumn('FIREPLACES', df['FIREPLACES'].cast('double'))
# Create the Binarizer transformer (threshold=0.0 i.e. over 0 assigned 1)
bin = Binarizer(threshold=0.0, inputCol='FIREPLACES', outputCol='FireplaceT')
# Apply transformer
df = bin.transform(df)

df[['FIREPLACES', 'FireplaceT']].show(3)

## Bucketing (Creating ordinal values)

from pyspark.ml.feature import Bucketizer
# Define how to split data using specified splits
splits = [0, 1, 2, 3, 4, float('Inf')]
# Create the Bucketizer transformer
buck = Bucketizer(splits=splits, inputCol='BATHSTOTAL', outputCol='baths')
# Apply transformer
df = buck.transform(df)
# Inspect the results
df[['BATHSTOTAL', 'baths']].show(4)

## One Hot Encoding (Pivot categorical value into indicator columns)

from pyspark.ml.feature import OneHotEncoder, StringIndexer

# Create indexer transformer
stringIndexer = StringIndexer(inputCol='CITY', outputCol='City_Index')

# Fit transformer
model = stringIndexer.fit(df)
# Apply transformer
indexed = model.transform(df)

# Create encoder transformer
encoder = OneHotEncoder(inputCol='City_Index', outputCol='City_Vec')
# Apply the encoder transformer
encoded_df = encoder.transform(indexed)
# Inspect results
encoded_df[['City_Vec']].show(4)

## Train-Test Splits for Time Series Data

# Create variables for max and min dates in our dataset
max_date = df.agg({'OFFMKTDATE': 'max'}).collect()[0][0]
min_date = df.agg({'OFFMKTDATE': 'min'}).collect()[0][0]

# Find how many days our data spans
from pyspark.sql.functions import datediff
range_in_days = datediff(max_date, min_date)

# Find the date to split the dataset on
from pyspark.sql.functions import date_add
split_in_days = round(range_in_days * 0.8)
split_date = date_add(min_date, split_in_days)

# Split the data into 80% train, 20% test
train_df = df.where(df['OFFMKTDATE'] < split_date)
test_df = df.where(df['OFFMKTDATE'] >= split_date).where(df['LISTDATE'] >= split_date)

## Data Preparation

# Check shape of data
print((df.count(), len(df.columns)))

from pyspark.ml.feature import VectorAssembler

# Replace Missing Values
df = df.fillna(-1) 
# Define columns to be converted to vectors
feature_cols = list(df.columns)
# Remove dependent variable
feature_cols.remove('SALESCLOSEPRICE')

# Create the vector assembler transformer
vec = VectorAssembler(inputCols=features_cols, outputCol='features')
# Apply the vector transformer to data
df = vec.transform(df)
# Select only the feature vectors and the dependent variable
ml_ready_df = df.select(['SALESCLOSEPRICE','features'])
# Inspect Results
ml_ready_df.show(5)

'''
+----------------+--------------------+
| SALESCLOSEPRICE| features|
+----------------+--------------------+
|143000 		|(125,[0,1,2,3,5,6...|
|190000 		|(125,[0,1,2,3,5,6...|
|225000 		|(125,[0,1,2,3,5,6...|
|265000 		|(125,[0,1,2,3,4,5...|
|249900 		|(125,[0,1,2,3,4,5...|
+----------------+--------------------+
only showing top 5 rows
'''

## Training Random Forest

from pyspark.ml.regression import RandomForestRegressor

# Initialize model with columns to utilize
rf = RandomForestRegressor(feature_cols = 'features',
							labelCol = 'SALESCLOSEPRICE',
							predictionCol = 'Prediction_Price',
							seed = 42
							)

# Train model
model = rf.fit(train_df)

# Generate Predictions
predictions = model.transform(test_df)

# Inspecting results
predictions.select("Prediction_Price", "SALESCLOSEPRICE").show(5)

## Model Evaluation
from pyspark.ml.evaluation import RegressionEvaluator

# Select columns to compute test error
evaluator = RegressionEvaluator(labelCol="SALESCLOSEPRICE", 
								predictionCol="Prediction_Price")

# Create evaluation metrics
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

# Print Model Metrics
print('RMSE: ' + str(rmse))
print('R^2: ' + str(r2))

## Model Interpretation

import pandas as pd

# Convert feature importances to Pandas column
fi_df = pd.DataFrame(model.featureImportances.toArray(),
						columns=['importance'])


# Convert list of feature names to Pandas column
fi_df['feature'] = pd.Series(feature_cols)

# Sort data based on feature importance
fi_df.sort_values(by=['importance'], ascending=False, inplace=True)

# Interpreting results
model_df.head(9)

## Saving & Loading Models

# Save model
model.save('rfr_real_estate_model')

from pyspark.ml.regression import RandomForestRegressionModel
# Load model from
model2 = RandomForestRegressionModel.load('rfr_real_estate_model')

## 1. Exploratory Data Analysis -------------------------

# Read the file into a dataframe
df = spark.read.parquet('Real_Estate.parq')
# Print columns in dataframe
print(df.columns)

# Select our dependent variable
Y_df = df.select(['SALESCLOSEPRICE'])
# Display summary statistics
Y_df.describe().show()

## Verify Data Load

def check_load(df, num_records, num_columns):
  # Takes a dataframe and compares record and column counts to input
  # Message to return if the critera below aren't met
  message = 'Validation Failed'
  # Check number of records
  if num_records == df.count():
    # Check number of columns
    if num_columns == len(df.columns):
      # Success message
      message = "Validation Passed"
  return message

# Print the data validation message
print(check_load(df, 5000, 74))

## Verifying DataTypes
# Validate dictionary of attributes `validation_dict` and their datatypes

# create list of actual dtypes to check
actual_dtypes_list = df.dtypes
print(actual_dtypes_list)

# Iterate through the list of actual dtypes tuples
for attribute_tuple in actual_dtypes_list:
  col_name = attribute_tuple[0]
  if col_name in validation_dict:   # Check if column name is dictionary of expected dtypes
    col_type = attribute_tuple[1]     # Compare attribute types
    if col_type == validation_dict[col_name]:
      print(col_name + ' has expected dtype.')

## Correlation Analysis

columns = df.columns

# Initialize to store name and value of col with max corr
corr_max = 0
corr_max_col = columns[0]

# Check correlation of all variables against dependent variable
for col in columns:
    corr_val = df.corr(col, 'SALESCLOSEPRICE')

    if corr_val > corr_max:
        # Update the column name and corr value
        corr_max = corr_val
        corr_max_col = col

print(corr_max_col)

## Distribution Plot

# Select a single column and sample and convert to pandas
sample_df = df.select(['LISTPRICE']).sample(False, .5, 42)
pandas_df = sample_df.toPandas()

# Plot distribution of pandas_df and display plot
sns.distplot(pandas_df)
plt.show() # Note: Observe a right-skewed distribution

# Import skewness function
from pyspark.sql.functions import skewness

# Compute and print skewness of LISTPRICE
print(df.agg({'LISTPRICE': 'skewness'}).collect())

## lmplot Plot

# Select a the relevant columns and sample
sample_df = df.select(['SALESCLOSEPRICE', 'LIVINGAREA']).sample(False, .5, 42)

# Convert to pandas dataframe
pandas_df = sample_df.toPandas()

# Linear model plot of pandas_df
sns.lmplot(x='LIVINGAREA', y='SALESCLOSEPRICE', data=pandas_df)
plt.show()

## 2. Wrangling with Spark Functions -------------------------

# Show top 30 records
df.show(30)

# Drop list of irrelevant columns
cols_to_drop = ['STREETNUMBERNUMERIC', 'LOTSIZEDIMENSIONS']
df = df.drop(*cols_to_drop)

## Applying text filter to remove records

# Inspect unique values in the column 'ASSUMABLEMORTGAGE'
df.select(['ASSUMABLEMORTGAGE']).distinct().show()
# List of possible values containing 'yes'
yes_values = ['Yes w/ Qualifying', 'Yes w/No Qualifying']
# Filter the text values out of df but keep null values
text_filter = ~df['ASSUMABLEMORTGAGE'].isin(yes_values) | df['ASSUMABLEMORTGAGE'].isNull()
df = df.where(text_filter)
# Print count of remaining records
print(df.count())

## Filtering numeric fields conditionally

from pyspark.sql.functions import mean, stddev

# Calculate values used for outlier filtering
mean_val = df.agg({'log_SalesClosePrice': 'mean'}).collect()[0][0]
stddev_val = df.agg({'log_SalesClosePrice': 'stddev'}).collect()[0][0]

# Create three standard deviation (μ ± 3σ) lower and upper bounds for data
low_bound = mean_val - (3 * stddev_val)
hi_bound = mean_val + (3 * stddev_val)

# Filter the data to fit between the lower and upper bounds
df = df.where((df['log_SalesClosePrice'] < hi_bound) & (df['log_SalesClosePrice'] > low_bound))

## Custom Percentage Scaling with minmax

# Define max and min values and collect them
max_days = df.agg({'DAYSONMARKET': 'max'}).collect()[0][0]
min_days = df.agg({'DAYSONMARKET': 'min'}).collect()[0][0]

# Create a new column based off the scaled data
df = df.withColumn('percentagescaleddays', round((df['DAYSONMARKET'] - min_days) / (max_days - min_days)) * 100)

# Calc max and min for new column
print(df.agg({'percentagescaleddays': 'max'}).collect())
print(df.agg({'percentagescaleddays': 'min'}).collect())

## Generalizing scaler for multiple columns

def min_max_scaler(df, cols_to_scale):
  
  # Takes a dataframe and list of columns to minmax scale. Returns a dataframe.
  
  for col in cols_to_scale:
    
    # Define min and max values and collect them
    max_days = df.agg({col: 'max'}).collect()[0][0]
    min_days = df.agg({col: 'min'}).collect()[0][0]
    new_column_name = 'scaled_' + col
    
    # Create a new column based off the scaled data
    df = df.withColumn(new_column_name, 
                      (df[col] - min_days) / (max_days - min_days))
    return df

cols_to_scale = ['FOUNDATIONSIZE', 'DAYSONMARKET', 'FIREPLACES']
df = min_max_scaler(df, cols_to_scale)

# Check data scaled between 0 and 1
df[['DAYSONMARKET', 'scaled_DAYSONMARKET']].show()

## Correcting Right Skew Data (i.e. Negative Skew)
# Note for right skew, need to "Reflect" data first using the following formula:
# (x_max + 1) - x

from pyspark.sql.functions import log

# Compute the skewness
print(df.agg({'YEARBUILT': 'skewness'}).collect())
# Calculate the max year
max_year = df.agg({'YEARBUILT': 'max'}).collect()[0][0]

# Create a new column of reflected data
df = df.withColumn('Reflect_YearBuilt', (max_year + 1) - df['YEARBUILT'])
# Create a new column based reflected data
df = df.withColumn('adj_yearbuilt', 1 / log(df['Reflect_YearBuilt']))

## Visualizing Missing Data

# Sample the dataframe and convert to Pandas
sample_df = df.select(columns).sample(False, 0.1, 42)
pandas_df = sample_df.toPandas()

# Convert all values to T/F
tf_df = pandas_df.isnull()

# Plot it
sns.heatmap(data=tf_df)
plt.xticks(rotation=30, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.show()

# Identifying column with most missing data
most_missing_col = 'BACKONMARKETDATE'

## Imputing Missing Data

# Count missing rows
df.where(df['PDOM'].isNull()).count()

# Impute missing data with column mean
col_mean = df.agg({'PDOM': 'mean'}).collect()[0][0]
df.fillna(col_mean, subset=['PDOM'])

## Calculating Missing Percents with Thresholding

def column_dropper(df, threshold):
  # Takes a dataframe and threshold for missing values. Returns a dataframe.
  total_records = df.count()
  for col in df.columns:
    # Calculate the percentage of missing values
    missing = df.where(df[col].isNull()).count()
    missing_percent = missing / total_records
    # Drop column if percent of missing is more than threshold
    if missing_percent > threshold:
      df = df.drop(col)
  return df

# Drop columns that are more than 60% missing
df = column_dropper(df, .6)

## Joining on same precision (decimal digits)
# Important to ensure that join keys are in the same format and precision!

# Cast data types
walk_df = walk_df.withColumn('longitude', walk_df['longitude'].cast('double'))
walk_df = walk_df.withColumn('latitude', walk_df['latitude'].cast('double'))

# Round precision
df = df.withColumn('longitude', round(df['longitude'], 5))
df = df.withColumn('latitude', round(df['latitude'], 5))

# Create join condition
condition = [df['latitude'] == walk_df['latitude'], df['longitude'] == walk_df['longitude']]

# Join the dataframes together
join_df = df.join(walk_df, on=condition, how='left')
# Count non-null records from new field
print(join_df.where(~join_df['walkscore'].isNull()).count())

## SparkSQL Join

# Register dataframes as tables
df.createOrReplaceTempView("df")
walk_df.createOrReplaceTempView("walk_df")

# SQL to join dataframes
join_sql = 	"""
			SELECT 
				*
			FROM df
			LEFT JOIN walk_df
			ON df.longitude = walk_df.longitude
			AND df.latitude = walk_df.latitude
			"""
# Perform sql join
joined_df = spark.sql(join_sql)

## Checking for Bad Joins

# Join on mismatched keys precision 
wrong_prec_cond = [df_orig['longitude'] == walk_df['longitude'], df_orig['latitude'] == walk_df['latitude']]
wrong_prec_df = df_orig.join(walk_df, on=wrong_prec_cond, how='left')

# Compare bad join to the correct one
print(wrong_prec_df.where(wrong_prec_df['walkscore'].isNull()).count())
print(correct_join_df.where(correct_join_df['walkscore'].isNull()).count())

# 4999
# 151

# Create a join on too few keys
few_keys_cond = [df['longitude'] == walk_df['longitude']]
few_keys_df = df.join(walk_df, on=few_keys_cond, how='left')

# Compare bad join to the correct one
print("Record Count of the Too Few Keys Join Example: " + str(few_keys_df.count()))
print("Record Count of the Correct Join Example: " + str(correct_join_df.count()))

# Record Count of the Too Few Keys Join Example: 6152
# Record Count of the Correct Join Example: 5000

# In Summary:
# Incorrect join conditions can lead to corrupted data (more entries than intended, high null values etc.)

## 3. Feature Engineering -------------------------

## Differencing
# Not a lot of variation in lot sizes, want to create strong features

# Lot size in square feet
acres_to_sqfeet = 43560
df = df.withColumn('LOT_SIZE_SQFT', df['ACRES'] * acres_to_sqfeet)

# Create new column YARD_SIZE
df = df.withColumn('YARD_SIZE', df['LOT_SIZE_SQFT'] - df['FOUNDATIONSIZE'])

# Corr of ACRES vs SALESCLOSEPRICE
print("Corr of ACRES vs SALESCLOSEPRICE: " + str(df.corr('ACRES', 'SALESCLOSEPRICE')))
# Corr of FOUNDATIONSIZE vs SALESCLOSEPRICE
print("Corr of FOUNDATIONSIZE vs SALESCLOSEPRICE: " + str(df.corr('FOUNDATIONSIZE', 'SALESCLOSEPRICE')))
# Corr of YARD_SIZE vs SALESCLOSEPRICE
print("Corr of YARD_SIZE vs SALESCLOSEPRICE: " + str(df.corr('YARD_SIZE', 'SALESCLOSEPRICE')))

## Ratios

# ASSESSED_TO_LIST
df = df.withColumn('ASSESSED_TO_LIST', df['ASSESSEDVALUATION'] / df['LISTPRICE'])
df[['ASSESSEDVALUATION', 'LISTPRICE', 'ASSESSED_TO_LIST']].show(5)
# TAX_TO_LIST
df = df.withColumn('TAX_TO_LIST', df['TAXES']/df['LISTPRICE'])
df[['TAX_TO_LIST', 'TAXES', 'LISTPRICE']].show(5)
# BED_TO_BATHS
df = df.withColumn('BED_TO_BATHS', df['BEDROOMS']/df['BATHSTOTAL'])
df[['BED_TO_BATHS', 'BEDROOMS', 'BATHSTOTAL']].show(5)

## Deeper Features (Combining variable effects)

# Create new feature by adding two features together
df = df.withColumn('Total_SQFT', df['SQFTBELOWGROUND'] + df['SQFTABOVEGROUND'])

# Create additional new feature using previously created feature
df = df.withColumn('BATHS_PER_1000SQFT', df['BATHSTOTAL'] / (df['Total_SQFT'] / 1000))
df[['BATHS_PER_1000SQFT']].describe().show()

# Sample and create pandas dataframe
pandas_df = df.sample(False, 0.5, 0).toPandas()

# Linear model plots
sns.jointplot(x='Total_SQFT', y='SALESCLOSEPRICE', data=pandas_df, kind="reg", stat_func=r2)
plt.show()
sns.jointplot(x='BATHS_PER_1000SQFT', y='SALESCLOSEPRICE', data=pandas_df, kind="reg", stat_func=r2)
plt.show()

'''
<script.py> output:
    +-------+-------------------+
    |summary| BATHS_PER_1000SQFT|
    +-------+-------------------+
    |  count|               5000|
    |   mean| 1.4302617483739894|
    | stddev|  14.12890410245937|
    |    min|0.39123630672926446|
    |    max|             1000.0|
    +-------+-------------------+

Observation of outlier from max value of 1000 bathrooms per 1000sqft.

Observation from jointplots() that `Total_SQFT` has better R2 value
than the alternative feature 'BATHS_PER_1000SQFT'
'''

## Feature Engineering on Time Components

# Import needed functions
from pyspark.sql.functions import to_date, dayofweek

# Convert to date type
df = df.withColumn('LISTDATE', to_date('LISTDATE'))

# Get the day of the week
df = df.withColumn('List_Day_of_Week', dayofweek('LISTDATE'))

# Sample and convert to pandas dataframe
sample_df = df.sample(False, .5, 42).toPandas()

# Plot count plot of of day of week
sns.countplot(x="List_Day_of_Week", data=sample_df)
plt.show()

# Note: PySpark convention of week 1 value (Sunday)

## Joining on Time Components

from pyspark.sql.functions import year

# Initialize dataframes
df = real_estate_df
price_df = median_prices_df

# Create year column
df = df.withColumn('list_year', year('LISTDATE'))

# Adjust year to match
df = df.withColumn('report_year', (df['list_year'] - 1))

# Create join condition
condition = [df['CITY'] == price_df['City'], df['report_year'] == price_df['Year']]

# Join the dataframes together
df = df.join(price_df, on=condition, how='left')
# Inspect that new columns are available
df[['MedianHomeValue']].show()

## Date Math (Creating lagged feature)

from pyspark.sql.functions import lag, datediff, to_date
from pyspark.sql.window import Window

# Cast data type
mort_df = mort_df.withColumn('DATE', to_date('DATE'))

# Create window
w = Window().orderBy(mort_df['DATE'])
# Create lag column
mort_df = mort_df.withColumn('DATE-1', lag('DATE', count=1).over(w))

# Calculate difference between date columns
mort_df = mort_df.withColumn('Days_Between_Report', datediff('DATE', 'DATE-1'))
# Print results
mort_df.select('Days_Between_Report').distinct().show()

## Extracting Text to New Features

# Inspect variable of interest
df.select('GARAGEDESCRIPTION').show(5, truncate = False)

'''
+--------------------------------------------------------------+
|GARAGEDESCRIPTION                                             |
+--------------------------------------------------------------+
|Attached Garage                                               |
|Attached Garage, Driveway - Asphalt, Garage Door Opener       |
|Attached Garage                                               |
|Attached Garage, Detached Garage, Tuckunder, Driveway - Gravel|
|Attached Garage, Driveway - Asphalt, Garage Door Opener       |
+--------------------------------------------------------------+
only showing top 5 rows

'''

# Import needed functions
from pyspark.sql.functions import when

# Create boolean conditions for string matches
has_attached_garage = df['GARAGEDESCRIPTION'].like('%Attached Garage%')
has_detached_garage = df['GARAGEDESCRIPTION'].like('%Detached Garage%')

# Conditional value assignment 
df = df.withColumn('has_attached_garage', when(has_attached_garage, 1)
                                          .when(has_detached_garage, 0)
                                          .otherwise(None))

# Inspect results
df[['GARAGEDESCRIPTION', 'has_attached_garage']].show(truncate=100)

## Splitting & Exploding

# Import needed functions
from pyspark.sql.functions import split, explode

# Convert string to list-like array
df = df.withColumn('garage_list', split(df['GARAGEDESCRIPTION'], ', '))
# Explode the values into new records
ex_df = df.withColumn('ex_garage_list', explode(df['garage_list']))
# Inspect the values
ex_df[['ex_garage_list']].distinct().show(100, truncate=50)

## Pivot & Join

from pyspark.sql.functions import coalesce, first

# Pivot 
piv_df = ex_df.groupBy('NO').pivot('ex_garage_list').agg(coalesce(first('constant_val')))

# Join the dataframes together and fill null
joined_df = df.join(piv_df, on='NO', how='left')

# List of Columns to zero fill
zfill_cols = piv_df.columns

# Zero fill the pivoted values
zfilled_df = joined_df.fillna(0, subset=zfill_cols)

## Binarizing Day of Week

# Import transformer
from pyspark.ml.feature import Binarizer

# Create the transformer
binarizer = Binarizer(threshold=5.0, inputCol='List_Day_of_Week', outputCol='Listed_On_Weekend')
# Apply the transformation to df
df = binarizer.transform(df)
# Verify transformation
df[['List_Day_of_Week', 'Listed_On_Weekend']].show()

## Bucketing (i.e. Splits)

from pyspark.ml.feature import Bucketizer

# Plot distribution of sample_df
sns.distplot(sample_df, axlabel='BEDROOMS')
plt.show()

# Create the bucket splits and bucketizer
splits = [0, 1, 2, 3, 4, 5, float('Inf')]
buck = Bucketizer(splits=splits, inputCol='BEDROOMS', outputCol='bedrooms')

# Apply the transformation to df: df_bucket
df_bucket = buck.transform(df)

# Display results
df_bucket[['BEDROOMS', 'bedrooms']].show()

## One Hot Encoding

from pyspark.ml.feature import OneHotEncoder, StringIndexer

# Map strings to numbers with string indexer
string_indexer = StringIndexer(inputCol='SCHOOLDISTRICTNUMBER', outputCol='School_Index')
indexed_df = string_indexer.fit(df).transform(df)

# Onehot encode indexed values
encoder = OneHotEncoder(inputCol='School_Index', outputCol='School_Vec')
encoded_df = encoder.transform(indexed_df)

# Inspect the transformation steps
encoded_df[['SCHOOLDISTRICTNUMBER', 'School_Index', 'School_Vec']].show(truncate=100)

## 4. Building a Model -------------------------

## Train-Test Split for Time Series Data

def train_test_split_date(df, split_col, test_days=45):
  """Calculate the date to split test and training sets"""
  # Find how many days our data spans
  max_date = df.agg({split_col: 'max'}).collect()[0][0]
  min_date = df.agg({split_col: 'min'}).collect()[0][0]
  # Subtract an integer number of days from the last date in dataset
  split_date = max_date - timedelta(days=test_days)
  return split_date

# Find the date to use in spitting test and train
split_date = train_test_split_date(df, 'OFFMKTDATE')

# Create Sequential Test and Training Sets
train_df = df.where(df['OFFMKTDATE'] < split_date) 
test_df = df.where(df['OFFMKTDATE'] >= split_date).where(df['LISTDATE'] <= split_date) 

## Adjusting Time Features for Data Leakage

from pyspark.sql.functions import datediff, to_date, lit

split_date = to_date(lit('2017-12-10'))
# Create Sequential Test set
test_df = df.where(df['OFFMKTDATE'] >= split_date).where(df['LISTDATE'] <= split_date)

# Create a copy of DAYSONMARKET to review later
test_df = test_df.withColumn('DAYSONMARKET_Original', test_df['DAYSONMARKET'])

# Recalculate DAYSONMARKET from what we know on our split date
test_df = test_df.withColumn('DAYSONMARKET', datediff(split_date, 'LISTDATE'))

# Review the difference
test_df[['LISTDATE', 'OFFMKTDATE', 'DAYSONMARKET_Original', 'DAYSONMARKET']].show()

## Dropping Columns with Low Observations

obs_threshold = 30
cols_to_remove = list()
# Inspect first 10 binary columns in list
for col in binary_cols[0:10]:
  # Count the number of 1 values in the binary column
  obs_count = df.agg({col: 'sum'}).collect()[0][0]
  # If less than our observation threshold, remove
  if obs_count <= obs_threshold:
    cols_to_remove.append(col)
    
# Drop columns and print starting and ending dataframe shapes
new_df = df.drop(*cols_to_remove)

print('Rows: ' + str(df.count()) + ' Columns: ' + str(len(df.columns)))
print('Rows: ' + str(new_df.count()) + ' Columns: ' + str(len(new_df.columns)))

# Comment:
# Removing low observation features can improve processing speed in training,
# prevent overfitting, improve interpretability

## Naive Handling for Missing and Categorical Values

# Replace missing values
df = df.fillna(-1, subset=['WALKSCORE', 'BIKESCORE'])

# Create list of StringIndexers using list comprehension
indexers = [StringIndexer(inputCol=col, outputCol=col+"_IDX")\
            .setHandleInvalid("keep") for col in categorical_cols]
# Create pipeline of indexers
indexer_pipeline = Pipeline(stages=indexers)
# Fit and Transform the pipeline to the original data
df_indexed = indexer_pipeline.fit(df).transform(df)

# Clean up redundant columns
df_indexed = df_indexed.drop(*categorical_cols)
# Inspect data transformations
print(df_indexed.dtypes)

## Building Regression Model

from pyspark.ml.regression import GBTRegressor

# Train a Gradient Boosted Trees (GBT) model.
gbt = GBTRegressor(featuresCol='features',
                           labelCol='SALESCLOSEPRICE',
                           predictionCol="Prediction_Price",
                           seed=42
                           )

# Train model.
model = gbt.fit(train_df)

## Evaluating & Comparing Algorithms

from pyspark.ml.evaluation import RegressionEvaluator

# Select columns to compute test error
evaluator = RegressionEvaluator(labelCol='SALESCLOSEPRICE', 
                                predictionCol='Prediction_Price')
# Dictionary of model predictions to loop over
models = {'Gradient Boosted Trees': gbt_predictions, 'Random Forest Regression': rfr_predictions}
for key, preds in models.items():
  # Create evaluation metrics
  rmse = evaluator.evaluate(preds, {evaluator.metricName: "rmse"})
  r2 = evaluator.evaluate(preds, {evaluator.metricName: "r2"})
  
  # Print Model Metrics
  print(key + ' RMSE: ' + str(rmse))
  print(key + ' R^2: ' + str(r2))

## Interpreting Results (Feature Importance)

# Convert feature importances to a pandas column
fi_df = pd.DataFrame(importances, columns=['importance'])

# Convert list of feature names to pandas column
fi_df['feature'] = pd.Series(feature_cols)

# Sort the data based on feature importance
fi_df.sort_values(by=['importance'], ascending=False, inplace=True)

# Inspect Results
fi_df.head(5)

'''
    importance             feature
36    0.256598          SQFT_TOTAL
4     0.212320               TAXES
6     0.166661          LIVINGAREA
5     0.094061  TAXWITHASSESSMENTS
3     0.074668     SQFTABOVEGROUND

'''
## Saving & Loading Models

from pyspark.ml.regression import RandomForestRegressionModel

# Save model
model.save('rfr_no_listprice')

# Load model
loaded_model = RandomForestRegressionModel.load('rfr_no_listprice')


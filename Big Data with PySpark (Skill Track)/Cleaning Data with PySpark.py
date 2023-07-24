## Cleaning Data with PySpark

## Introduction to Data Clearning with Apache Spark

# Defining Spark Schema
from pyspark.sql.types import *

peopleSchema = StructType([
  StructField('name', StringType(), True),
  StructField('age', IntegerType(), True),
  StructField('city', StringType(), True)
])

# Read CSV file containing data
people_df = spark.read.format('csv').load(name = 'rawdata.csv', schema = peopleSchema)

## Concept: Immutability and lazy processing
# Spark dataframes are immutable for for efficient share/create of new data representations throughout cluster

# Load the CSV file
aa_dfw_df = spark.read.format('csv').options(Header=True).load('AA_DFW_2018.csv.gz')
# Add the airport column using the F.lower() method
aa_dfw_df = aa_dfw_df.withColumn('airport', F.lower(aa_dfw_df['Destination Airport']))
# Drop the Destination Airport column
aa_dfw_df = aa_dfw_df.drop(aa_dfw_df['Destination Airport'])
# Show the DataFrame
aa_dfw_df.show()

# Parquet Format: Columnar data format, supported in Spark and other data processing frameworks.
# Supports predicate pushdown. Automatically stores schema information.

# Reading Parquet files
df = spark.read.format('parquet').load('filename.parquet')
df = spark.read.parquet('filename.parquet')

# Writing Parquet files
df.write.format('parquet').save('filename.parquet')
df.write.parquet('filename.parquet')

# For SparkSQL operations
df.createOrReplaceTempView('table')
table_df = spark.sql('SELECT * FROM table')

## Saving DataFrame in Parquet format

# View the row count of df1 and df2
print("df1 Count: %d" % df1.count())
print("df2 Count: %d" % df2.count())

# Combine the DataFrames into one
df3 = df1.union(df2)

# Save the df3 DataFrame in Parquet format
df3.write.parquet('AA_DFW_ALL.parquet', mode='overwrite')

# Read the Parquet file into a new DataFrame and run a count
print(spark.read.parquet('AA_DFW_ALL.parquet').count())

## SQL and Parquet

# Read the Parquet file into flights_df
flights_df = spark.read.parquet('AA_DFW_ALL.parquet')

# Register the temp table
flights_df.createOrReplaceTempView('flights')

# Run a SQL query of the average flight duration
avg_duration = spark.sql('SELECT avg(flight_duration) from flights').collect()[0]
print('The average flight time is: %d' % avg_duration)

## Manipulating DataFrames ----------------------------------

## DataFrame Column Operations

# Refresher on Common DataFrame Operations
voter_df.filter(voter_df.date > '1/1/2019')  # Filter rows where date is greater than '1/1/2019'
voter_df.select(voter_df.name)  # Select only the 'name' column
voter_df.withColumn('year', voter_df.date.year)  # Add a new static column named 'year' based on the 'date' column
voter_df.drop('column_to_drop')  # Drop the 'column_to_drop' column from the DataFrame

# Filtering Data - remove nulls, odd entries, split from combined sources
voter_df.filter(voter_df['name'].isNotNull())  # Filter rows where 'name' column is not null
voter_df.filter(voter_df.date.year > 1800)  # Filter rows where the year of the 'date' column is greater than 1800
voter_df.where(voter_df['_c0'].contains('VOTE'))  # Filter rows where the '_c0' column contains the substring 'VOTE'
voter_df.where(~voter_df._c1.isNull())  # Filter rows where the '_c1' column is not null using the negation operator (~)

## Exercise on Filtering Column content

# Show the distinct VOTER_NAME entries
voter_df.select(voter_df.VOTER_NAME).distinct().show(40, truncate=False)

# Filter voter_df where the VOTER_NAME is 1-20 characters in length
voter_df = voter_df.filter('length(VOTER_NAME) > 0 and length(VOTER_NAME) < 20')

# Filter out voter_df where the VOTER_NAME contains an underscore
voter_df = voter_df.filter(~ F.col('VOTER_NAME').contains("_"))

# Show the distinct VOTER_NAME entries again
voter_df.select(voter_df.VOTER_NAME).distinct().show(40, truncate=False)

## ---------- ##

# Column string transformations
import pyspark.sql.functions as F

voter_df.withColumn('upper', F.upper('name'))  # Add a new column named 'upper' that contains the uppercase values of the 'name' column
voter_df.withColumn('splits', F.split('name', ' '))  # Add a new column named 'splits' that contains an array of strings split by space from the 'name' column
voter_df.withColumn('year', voter_df['_c4'].cast(IntegerType()))  # Add a new column named 'year' by casting the '_c4' column to IntegerType()

# Additional ArrayType() column functions
# .size(<column>) - return length of ArrayType() column
# .getItem(<index>) - retrieve specific item at index of list column

## Exercise on Modifying DataFrame columns

# Add a new column called splits separated on whitespace
voter_df = voter_df.withColumn("splits", F.split(voter_df.VOTER_NAME, '\s+'))
# Create a new column called first_name based on the first item in splits
voter_df = voter_df.withColumn("first_name", voter_df.splits.getItem(0))
# Get the last entry of the splits list and create a column called last_name
voter_df = voter_df.withColumn("last_name", voter_df.splits.getItem(F.size('splits') - 1))
# Drop the splits column
voter_df = voter_df.drop('splits')
# Show the voter_df DataFrame
voter_df.show()

## Conditional DataFrame column operations

# Syntax: .when(<if condition>, <then x>)

df.select(df.Name, df.Age, F.when(df.Age >= 18, "Adult"))

df.select(df.Name, df.Age,
	F.when(df.Age >= 18, "Adult")
	.when(df.Age < 18, "Minor"))
# Equivalently:
df.select(df.Name, df.Age,
	F.when(df.Age >= 18, "Adult")
	.otherwise("Minor"))

## Example:
# Add a column to voter_df for any voter with the title "Councilmember"
voter_df = voter_df.withColumn('random_val',
                               when(voter_df.TITLE == "Councilmember", F.rand()))

# Show some of the DataFrame rows, noting whether the when clause worked
voter_df.show(5)

## Example:

# Add a column to voter_df for a voter based on their position
voter_df = voter_df.withColumn('random_val',
                               when(voter_df.TITLE == 'Councilmember', F.rand())
                               .when(voter_df.TITLE == "Mayor", 2)
                               .otherwise(0))

# Show some of the DataFrame rows
voter_df.show()

# Use the .filter() clause with random_val
voter_df.filter(voter_df.random_val == 0).show()

## User-Defined Functions (UDFs)

# Given a defined python method:
def reverseString(mystr):
	return mystr[::-1]

udfReverseString = udf(reverseString, StringType()) # wrap and store using udf

# Use with Spark
user_df = user_df.withColumn('ReverseName',
	udfReverseString(user_df.Name))

# Another Example: Function call without arguments
def sortingCap():
	return random.choice(['G','H','R','S'])
udfSortingCap = udf(sortingCap,StringType())
user_df = user_df.withColumn('Class', udfSortingCap())

## Exercise:

def getFirstAndMiddle(names):
  # Return a space separated string of names
  return ' '.join(names)

# Define the method as a UDF
udfFirstAndMiddle = F.udf(getFirstAndMiddle, StringType())

# Create a new column using your UDF
voter_df = voter_df.withColumn('first_and_middle_name', udfFirstAndMiddle(voter_df.splits))

# Show the DataFrame
voter_df.show()

## Partitioning and Lazy Processing

pyspark.sql.functions.monotonically_increasing_id() # to allow for completely parallel processing of IDs that are still unique

# Notes: Spark is lazy, occasionally out of order, e.g ID may be assigned after join. Test transformations.

## Exercise: Adding ID Field

# Select all the unique council voters
voter_df = df.select(df["VOTER NAME"]).distinct()
# Count the rows in voter_df
print("\nThere are %d rows in the voter_df DataFrame.\n" % voter_df.count())

# Add a ROW_ID
voter_df = voter_df.withColumn('ROW_ID', F.monotonically_increasing_id())

# Show the rows with 10 highest IDs in the set
voter_df.orderBy(voter_df.ROW_ID.desc()).show(10)

## Exercise: IDs with different partitions

# Print the number of partitions in each DataFrame
print("\nThere are %d partitions in the voter_df DataFrame.\n" % voter_df.rdd.getNumPartitions())
print("\nThere are %d partitions in the voter_df_single DataFrame.\n" % voter_df_single.rdd.getNumPartitions())

# Add a ROW_ID field to each DataFrame
voter_df = voter_df.withColumn('ROW_ID', F.monotonically_increasing_id())
voter_df_single = voter_df_single.withColumn('ROW_ID', F.monotonically_increasing_id())

# Show the top 10 IDs in each DataFrame 
voter_df.orderBy(voter_df.ROW_ID.desc()).show(10)
voter_df_single.orderBy(voter_df_single.ROW_ID.desc()).show(10)

## Exercise: More ID tricks - start IDs from value

# Determine the highest ROW_ID and save it in previous_max_ID
previous_max_ID = voter_df_march.select('ROW_ID').rdd.max()[0]

# Add a ROW_ID column to voter_df_april starting at the desired value
voter_df_april = voter_df_april.withColumn('ROW_ID', previous_max_ID + F.monotonically_increasing_id())

# Show the ROW_ID from both DataFrames and compare
voter_df_march.select('ROW_ID').show()
voter_df_april.select('ROW_ID').show()

## ----------------------------------

## Implementing caching
# Store DataFrame in memory or disk, improve speed on subsequent transformations/actions. 
# Reduce resource usage. Note very large datasets may not fit in memory.

voter_df = spark.read.csv("voter_data.txt.gz")
voter_df.cache().count()

# Add a new column 'ID' with monotonically increasing IDs, cache in memory for faster access
voter_df = voter_df.withColumn('ID', monotonically_increasing_id())
voter_df = voter_df.cache()
voter_df.show()

# Check cache status
print(voter_df.is_cached)

# Remove all blocks from memory and disk when finished with DataFrame
voter_df.unpersist()

## Check time for cached DataFrame

start_time = time.time()

# Add caching to the unique rows in departures_df
departures_df = departures_df.distinct().cache()

# Count the unique rows in departures_df, noting how long the operation takes
print("Counting %d rows took %f seconds" % (departures_df.count(), time.time() - start_time))

# Count the rows again, noting the variance in time of a cached DataFrame
start_time = time.time()
print("Counting %d rows again took %f seconds" % (departures_df.count(), time.time() - start_time))

## Removing DataFrame from cache

# Determine if departures_df is in the cache
print("Is departures_df cached?: %s" % departures_df.is_cached)
print("Removing departures_df from cache")

# Remove departures_df from the cache
departures_df.unpersist()

# Check the cache status again
print("Is departures_df cached?: %s" % departures_df.is_cached)

## File Import Performance

# Import the full and split files into DataFrames
full_df = spark.read.csv('departures_full.txt.gz')
split_df = spark.read.csv('departures_000.txt.gz')

# Print the count and run time for each DataFrame
start_time_a = time.time()
print("Total rows in full DataFrame:\t%d" % full_df.count())
print("Time to run: %f" % (time.time() - start_time_a))

start_time_b = time.time()
print("Total rows in split DataFrame:\t%d" % split_df.count())
print("Time to run: %f" % (time.time() - start_time_b))

## Reading Spark Configurations

# Name of the Spark application instance
app_name = spark.conf.get('spark.app.name')

# Driver TCP port
driver_tcp_port = spark.conf.get('spark.driver.port')

# Number of join partitions
num_partitions = spark.conf.get('spark.sql.shuffle.partitions')

# Show the results
print("Name: %s" % app_name)
print("Driver TCP port: %s" % driver_tcp_port)
print("Number of partitions: %s" % num_partitions)

## Normal Joins

# Join the flights_df and aiports_df DataFrames
normal_df = flights_df.join(airports_df, \
    flights_df["Destination Airport"] == airports_df["IATA"] )

# Show the query plan
normal_df.explain()

## Using broadcasting on Spark joins
# Import the broadcast method from pyspark.sql.functions
from pyspark.sql.functions import broadcast

# Join the flights_df and airports_df DataFrames using broadcasting
broadcast_df = flights_df.join(broadcast(airports_df), \
    flights_df["Destination Airport"] == airports_df["IATA"] )

# Show the query plan and compare against the original
broadcast_df.explain()

## Comparison of performance

start_time = time.time()
# Count the number of rows in the normal DataFrame
normal_count = normal_df.count()
normal_duration = time.time() - start_time

start_time = time.time()
# Count the number of rows in the broadcast DataFrame
broadcast_count = broadcast_df.count()
broadcast_duration = time.time() - start_time

# Print the counts and the duration of the tests
print("Normal count:\t\t%d\tduration: %f" % (normal_count, normal_duration))
print("Broadcast count:\t%d\tduration: %f" % (broadcast_count, broadcast_duration))

## ----------------------------------

## Complex Processing and Data Pipelines

## Quick Pipeline

# Import the data to a DataFrame
departures_df = spark.read.csv('2015-departures.csv.gz', header=True)
# Remove any duration of 0
departures_df = departures_df.filter(departures_df["Actual elapsed time (Minutes)"] > 0)
# Add an ID column
departures_df = departures_df.withColumn('id', F.monotonically_increasing_id())
# Write the file out to JSON format
departures_df.write.json('output.json', mode='overwrite')

## Removing commented lines

# Import the file to a DataFrame and perform a row count
annotations_df = spark.read.csv('annotations.csv.gz', sep="|")
full_count = annotations_df.count()

# Count the number of rows beginning with '#'
comment_count = annotations_df.filter(col('_c0').startswith('#')).count()

# Import the file to a new DataFrame, without commented rows
no_comments_df = spark.read.csv('annotations.csv.gz', sep="|", comment='#')

# Count the new DataFrame and verify the difference is as expected
no_comments_count = no_comments_df.count()
print("Full count: %d\nComment count: %d\nRemaining count: %d" % (full_count, comment_count, no_comments_count))

## Removing Invalid Rows:

# Split _c0 on the tab character and store the list in a variable
tmp_fields = F.split(annotations_df['_c0'], '\t')

# Create the colcount column on the DataFrame
# Note: F.size() returns no. of elements in array/no. characters in string column
annotations_df = annotations_df.withColumn('colcount', F.size(tmp_fields))

# Remove any rows containing fewer than 5 fields
annotations_df_filtered = annotations_df.filter(~ (annotations_df.colcount >= 5))

# Count the number of rows
final_count = annotations_df_filtered.count()
print("Initial count: %d\nFinal count: %d" % (initial_count, final_count))

## Splitting into columns

# Split the content of _c0 on the tab character (aka, '\t')
split_cols = F.split(annotations_df['_c0'], '\t')

# Add the columns folder, filename, width, and height
split_df = annotations_df.withColumn('folder', split_cols.getItem(0))
split_df = split_df.withColumn('filename', split_cols.getItem(1))
split_df = split_df.withColumn('width', split_cols.getItem(2))
split_df = split_df.withColumn('height', split_cols.getItem(3))

# Add split_cols as a column
split_df = split_df.withColumn('split_cols', split_cols)

## Further Parsing

def retriever(cols, colcount):
  # Return a list of dog data
  return cols[4:colcount]

# Define the method as a UDF
udfRetriever = F.udf(retriever, ArrayType(StringType()))

# Create a new column using your UDF
split_df = split_df.withColumn('dog_list', udfRetriever(split_df.split_cols, split_df.colcount))

# Remove the original column, split_cols, and the colcount
split_df = split_df.drop('_c0').drop('split_cols').drop('colcount')

##  // Data Validation //

## Validate Rows via Join

# Rename the column in valid_folders_df
valid_folders_df = valid_folders_df.withColumnRenamed("_c0","folder")
# Count the number of rows in split_df
split_count = split_df.count()
# Join the DataFrames
joined_df = split_df.join(F.broadcast(valid_folders_df), "folder")
# Compare the number of rows remaining
joined_count = joined_df.count()
print("Before: %d\nAfter: %d" % (split_count, joined_count))

## Examining Invalid Rows

# Determine the row counts for each DataFrame
split_count = split_df.count()
joined_count = joined_df.count()

# Create a DataFrame containing the invalid rows
invalid_df = split_df.join(F.broadcast(joined_df), 'filename', 'left_anti')

# Validate the count of the new DataFrame is as expected
invalid_count = invalid_df.count()
print(" split_df:\t%d\n joined_df:\t%d\n invalid_df: \t%d" % (split_count, joined_count, invalid_count))

# Determine the number of distinct folder rows removed
invalid_folder_count = invalid_df.select('folder').distinct().count()
print("%d distinct invalid folders found" % invalid_folder_count)

## // Final Analysis and Delivery //

def getAvgSale(saleslist):
    totalsales = 0
    count = 0
    for sale in saleslist:
        totalsales += sale[2] + sale[3]
        count += 2
    return totalsales / count

udfGetAvgSale = udf(getAvgSale, DoubleType())
df = df.withColumn('avg_sale', udfGetAvgSale(df.sales_list))

# Inline calculations
df = spark.read.csv('datafile')
df = df.withColumn('avg', (df['total_sales'] / df['sales_count'])) # Calculate the 'avg' column
df = df.withColumn('sq_ft', df['width'] * df['length']) # Calculate the 'sq_ft' column
df = df.withColumn('total_avg_size', udfComputeTotal(df.entries) / df.numEntries) # Using UDF

## Dog Parsing -------------------------

# Select the dog details and show 10 untruncated rows
print(joined_df.select("dog_list").show(10, truncate=False))

# Define a schema type for the details in the dog list
DogType = StructType([
	StructField("breed", StringType(), False),
    StructField("start_x", IntegerType(), False),
    StructField("start_y", IntegerType(), False),
    StructField("end_x", IntegerType(), False),
    StructField("end_y", IntegerType(), False)
])

# Create a function to return the number and type of dogs as a tuple
def dogParse(doglist):
  dogs = []
  for dog in doglist:
    (breed, start_x, start_y, end_x, end_y) = dog.split(',')
    dogs.append((breed, int(start_x), int(start_y), int(end_x), int(end_y)))
  return dogs

# Create a UDF
udfDogParse = F.udf(dogParse, ArrayType(DogType))

# Use the UDF to list of dogs and drop the old column
joined_df = joined_df.withColumn('dogs', udfDogParse('dog_list')).drop('dog_list')

# Show the number of dogs in the first 10 rows
joined_df.select(F.size('dogs')).show(10)

# Define a UDF to determine the number of pixels per image
def dogPixelCount(doglist):
  totalpixels = 0
  for dog in doglist:
    totalpixels += (dog[3] - dog[1]) * (dog[4] - dog[2])
  return totalpixels

# Define a UDF for the pixel count
udfDogPixelCount = F.udf(dogPixelCount, IntegerType())
joined_df = joined_df.withColumn('dog_pixels', udfDogPixelCount('dogs'))

# Create a column representing the percentage of pixels
joined_df = joined_df.withColumn('dog_percent', (joined_df.dog_pixels / (joined_df.width * joined_df.height)) * 100)

# Show the first 10 annotations with more than 60% dog
joined_df.where('dog_percent > 60').show(10)



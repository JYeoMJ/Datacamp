# Cleaning Data with PySpark

Data cleaning is a crucial step in any data analysis pipeline. In the PySpark environment, we have several techniques and functionalities available to make this process efficient and straightforward. This tutorial/reference guide will cover some of the essential techniques used in PySpark for data cleaning and transformation.

## Introduction to Data Cleaning with Apache Spark
***
Spark Schemas define the format of a DataFrame, which you can get by calling the `printSchema()` method on the DataFrame object. Spark SQL provides `StructType` & `StructField` classes to programmatically specify the schema.

Here's an example of defining a Spark schema using `StructType` and `StructField`:

```python
import pyspark.sql.types

# Define Schema
peopleSchema = StructType([
  # StructField(<name:String>, <DataType>, <nullable:Boolean>)
  StructField('name', StringType(), True),
  StructField('age', IntegerType(), True),
  StructField('city', StringType(), True)
])

# Read CSV file using defined schema
people_df = spark.read.format('csv').load(name = 'rawdata.csv', schema = peopleSchema)
```

### Concept: Immutability and Lazy Evaluation

One of the key concepts in Spark is the idea of immutability and lazy evaluation. Immutability means that once a DataFrame is created, it cannot be changed. Instead, any operation that appears to change it actually creates a new DataFrame. Lazy evaluation means that computation is delayed until an action is triggered.

Spark DataFrames are immutable for for efficient share/create of new data representations throughout cluster

```python
# Load the CSV file
aa_dfw_df = spark.read.format('csv').options(Header=True).load('AA_DFW_2018.csv.gz')
# Add the airport column using the F.lower() method
aa_dfw_df = aa_dfw_df.withColumn('airport', F.lower(aa_dfw_df['Destination Airport']))
# Drop the Destination Airport column
aa_dfw_df = aa_dfw_df.drop(aa_dfw_df['Destination Airport'])
# Show the DataFrame
aa_dfw_df.show()
```

### Parquet Format

[Parquet](https://parquet.apache.org/) is a columnar format that is supported by many other data processing systems. 

Spark SQL provides support for both reading and writing Parquet files that automatically preserves the schema of the original data. When reading Parquet files, all columns are automatically converted to be nullable for compatibility reasons.

```python
# Reading Parquet files
df = spark.read.format('parquet').load('filename.parquet')
df = spark.read.parquet('filename.parquet')

# Writing Parquet files
df.write.format('parquet').save('filename.parquet')
df.write.parquet('filename.parquet')

# Parquet files can also be used to create a temporary view and
# then used in SQL statements
df.createOrReplaceTempView('table')
table_df = spark.sql('SELECT * FROM table')
table_df.show()
```

Consider the following example:

```python
## // Saving DataFrame in Parquet format //

# View the row count of df1 and df2
print("df1 Count: %d" % df1.count())
print("df2 Count: %d" % df2.count())

# Combine the DataFrames into one
df3 = df1.union(df2)
# Save the df3 DataFrame in Parquet format
df3.write.parquet('AA_DFW_ALL.parquet', mode='overwrite')
# Read the Parquet file into a new DataFrame and run a count
print(spark.read.parquet('AA_DFW_ALL.parquet').count())

## // SQL and Parquet //
# Read the Parquet file into flights_df
flights_df = spark.read.parquet('AA_DFW_ALL.parquet')
# Register the temp table
flights_df.createOrReplaceTempView('flights')
# Run a SQL query of the average flight duration
avg_duration = spark.sql('SELECT avg(flight_duration) from flights').collect()[0]
print('The average flight time is: %d' % avg_duration)
```

## Manipulating DataFrames
***
### Refresher on Common DataFrame Operations

In PySpark, we often need to select specific columns or filter rows based on certain conditions. Below are some basic operations for selection and filtering.

```python
# Filter rows where date is greater than '1/1/2019'
voter_df.filter(voter_df.date > '1/1/2019')
# Select only the 'name' column
voter_df.select(voter_df.name)  
# Add a new static column named 'year' based on the 'date' column
voter_df.withColumn('year', voter_df.date.year)
# Drop the 'column_to_drop' column from the DataFrame
voter_df.drop('column_to_drop')
```

### Filtering Data

```python
# Filter rows where 'name' column is not null
voter_df.filter(voter_df['name'].isNotNull())  
# Filter rows where the year of the 'date' column is greater than 1800
voter_df.filter(voter_df.date.year > 1800)  
# Filter rows where the '_c0' column contains the substring 'VOTE'
voter_df.where(voter_df['_c0'].contains('VOTE'))  
# Filter rows where '_c1' column is not null using negation operator (~)
voter_df.where(~voter_df._c1.isNull())  
```

In the above code blocks, we are using the `select`, `filter`, and `where` methods provided by PySpark. `select` is used to pick specific columns from the DataFrame, while `filter` and `where` (which are aliases and work the same way) are used to filter rows based on specific conditions.

For instance, `voter_df.filter(voter_df.date.year > 1800)` filters the rows where the year part of the 'date' column is greater than 1800. The tilde symbol `~` before `voter_df._c1.isNull()` is a negation operator, meaning it filters the rows where the `_c1` column is not null.

### Advanced Selection and Filtering

In some scenarios, we need more advanced ways to select or filter data. The following examples demonstrate some of these techniques:

```python
# Show the distinct VOTER_NAME entries
voter_df.select(voter_df.VOTER_NAME).distinct().show(40, truncate=False)
# Filter voter_df where the VOTER_NAME is 1-20 characters in length
voter_df = voter_df.filter('length(VOTER_NAME) > 0 and length(VOTER_NAME) < 20')
# Filter out voter_df where the VOTER_NAME contains an underscore
voter_df = voter_df.filter(~ F.col('VOTER_NAME').contains("_"))
# Show the distinct VOTER_NAME entries again
voter_df.select(voter_df.VOTER_NAME).distinct().show(40, truncate=False)
```

In these examples, `distinct` is used to select unique rows, and the `filter` function is used with a string condition to filter rows based on the length of the 'VOTER_NAME'. We also use `F.col` to refer to the column inside `filter`.

### Column String Transformations

We often need to transform or create new columns based on the existing ones. PySpark provides several functions to do this. Here are some examples:

```python
import pyspark.sql.functions as F

# Add new column named 'upper' containing uppercase values of 'name'
voter_df.withColumn('upper', F.upper('name'))
# New column 'splits' with array of strings split by space from 'name'
voter_df.withColumn('splits', F.split('name', ' ')) 
# New column 'year' by casting the '_c4' column to IntegerType()
voter_df.withColumn('year', voter_df['_c4'].cast(IntegerType()))
```

The `withColumn` function is used to add a new column or transform an existing one. `F.upper` is used to convert the string to uppercase, `F.split` is used to split the string into an array of substrings, and `cast` is used to change the data type of a column.

### ArrayType Column Functions

For `ArrayType` column, PySpark provides functions to manipulate and retrieve information from it:
- `.size(<column>)` - returns the length of ArrayType() column
- `.getItem(<index>)` - retrieves a specific item at the index of the list column

Here's how you can use these functions:

```python
# Add a new column called splits separated on whitespace
voter_df = voter_df.withColumn("splits", F.split(voter_df.VOTER_NAME, '\s+'))
# Create a new column called first_name based on the first item in splits
voter_df = voter_df.withColumn("first_name", voter_df.splits.getItem(0))
# Get the last entry of the splits list and create a column called last_name
voter_df = voter_df.withColumn("last_name", voter_df.splits.getItem(F.size('splits') - 1))
# Drop the splits column
voter_df = voter_df.drop('splits')

# Display the DataFrame
voter_df.show()
```

In this code block, we first split the 'VOTER_NAME' into an array of words. We then create new columns for the first and last words in the array using `getItem`. Finally, we drop the 'splits' column as it's no longer needed.

### Conditional DataFrame Column Operations

PySpark supports a way to check multiple conditions in sequence and returns a value when conditions are met using the `.when()` expression

```python
# Define new (unamed) column with "Adult" label for ages >= 18
df.select(df.Name, df.Age, F.when(df.Age >= 18, "Adult"))

# Multiple .when() clauses
df.select(df.Name, df.Age,
	F.when(df.Age >= 18, "Adult")
	.when(df.Age < 18, "Minor"))
	
# Equivalently, using .otherwise()
df.select(df.Name, df.Age,
	F.when(df.Age >= 18, "Adult")
	.otherwise("Minor"))
```

The `when` function is used to check conditions, and the `otherwise` function is used to specify the output when none of the conditions in `when` are met.

### User-Defined Functions (UDFs)

User-Defined Functions (UDFs) in PySpark allow you to create your custom functions that extend the functionality of PySpark. These functions can then be applied directly on Spark DataFrames or Spark SQL operations. UDFs are a powerful feature that provides flexibility and makes it possible to carry out any complex computation that built-in functions cannot handle.

Let's see some examples:

```python
# Given a defined python method:
def reverseString(mystr):
	return mystr[::-1]

# wrap and store using udf method
udfReverseString = udf(reverseString, StringType())

# Use with Spark
user_df = user_df.withColumn('ReverseName',
	udfReverseString(user_df.Name))
```

In this example, we first define a Python function `reverseString` that reverses a string. We then convert this function into a UDF using `F.udf`. The `F.udf` function takes two parameters: the function and the return data type. Finally, we use this UDF in `withColumn` to apply the function on the 'Name' column, creating a new column 'ReverseName' with reversed strings.

UDFs can also accept multiple inputs, as shown in the following example:

```python
def getFirstAndMiddle(names):
  # Return a space separated string of names
  return ' '.join(names)

# Define the method as a UDF
udfFirstAndMiddle = F.udf(getFirstAndMiddle, StringType())

# Create a new column using your UDF
voter_df = voter_df.withColumn('first_and_middle_name', udfFirstAndMiddle(voter_df.splits))

# Show the DataFrame
voter_df.show()
```

In this example, the UDF `getFirstAndMiddle` accepts an array of names and joins them into a single string. This UDF is then used to create a new column 'first_and_middle_name' in the DataFrame.

In conclusion, UDFs are a vital tool in PySpark, allowing you to perform complex transformations and computations that are not readily available as built-in functions.

### Partitioning and Lazy Processing

Partitioning and lazy processing are two key concepts in PySpark that can significantly improve the performance of your data operations.

### Partitioning

Partitioning in PySpark is a way of dividing your data into smaller parts (partitions) that can be processed in parallel. Each partition is a collection of rows that sit on one physical machine in your cluster. PySpark automatically partitions your data and distributes the partitions across different nodes.

A function like `monotonically_increasing_id` can be used to assign unique IDs to each row in a DataFrame, even across partitions. This function generates a unique ID for each row, starting from 0.

```python
# Allow for completely parallel processing of IDs that are still unique
pyspark.sql.functions.monotonically_increasing_id()
```

The function `monotonically_increasing_id` is used to assign unique IDs to each row. It allows for completely parallel processing as each ID is unique across partitions.

### Lazy Processing

Lazy processing, or lazy evaluation, is a programming concept where the execution of operations is delayed until their results are needed. PySpark uses this concept to optimize the execution plan of data transformations. This means that when you perform a transformation operation in PySpark, it doesn't immediately execute the operation. Instead, it records the operation in a plan (known as a logical plan) and only executes it when an action (like count, show, etc.) is called.

For instance, consider the following example:

```python
# Select all the unique council voters
voter_df = df.select(df["VOTER NAME"]).distinct()
# Count the rows in voter_df
print("\nThere are %d rows in the voter_df DataFrame.\n" % voter_df.count())
# Add a ROW_ID
voter_df = voter_df.withColumn('ROW_ID', F.monotonically_increasing_id())
# Show the rows with 10 highest IDs in the set
voter_df.orderBy(voter_df.ROW_ID.desc()).show(10)
```

In this code block, the transformations `select`, `distinct`, and `withColumn` are not immediately executed. They are only executed when the actions `count` and `show` are called. This is an example of PySpark's lazy processing.

One thing to note about lazy processing is that the order of operations can sometimes be rearranged by the optimizer for efficiency. Therefore, it's important to understand this concept and test your transformations to ensure they are working as expected.

```python
## // More ID tricks - start IDs from value //
# Determine the highest ROW_ID and save it in previous_max_ID
previous_max_ID = voter_df_march.select('ROW_ID').rdd.max()[0]
# Add a ROW_ID column to voter_df_april starting at the desired value
voter_df_april = voter_df_april.withColumn('ROW_ID', previous_max_ID + F.monotonically_increasing_id())
# Show the ROW_ID from both DataFrames and compare
voter_df_march.select('ROW_ID').show()
voter_df_april.select('ROW_ID').show()
```

In the above example, `monotonically_increasing_id` is used to continue the ID sequence from `voter_df_march` to `voter_df_april`. This is done by finding the maximum ID in `voter_df_march` and adding it to the IDs in `voter_df_april`. Note that the `withColumn` operation is not immediately executed until `show` action is called.

By understanding and leveraging partitioning and lazy processing, you can optimize your PySpark operations for improved performance and efficiency.

## Improving Performance
***
When working with large datasets in PySpark, performance is a crucial aspect. There are several techniques you can use to improve the performance of your PySpark operations.

### Caching

Caching is a technique where the result of a transformation is stored in memory for quicker access during subsequent actions. This is especially useful when you are working with a DataFrame that is going to be accessed multiple times. Caching can help to reduce the time required to access data that is being reused, as it avoids re-computation of transformations applied on the DataFrame.

In Spark, when you call the `.cache()` method on a DataFrame, it tells Spark to store the DataFrame in memory after it is computed, but the actual caching does not occur until an action like `count()`, `show()`, etc., is called on the DataFrame. This is because Spark uses lazy evaluation, where the execution of operations is delayed until their results are needed.

```python
voter_df = spark.read.csv("voter_data.txt.gz")
voter_df.cache().count()

# Add a new column 'ID' with monotonically increasing IDs, cache in memory for faster access
voter_df = voter_df.withColumn('ID', monotonically_increasing_id())
voter_df = voter_df.cache()
voter_df.show()

# Check cache status
print(voter_df.is_cached)

# Remove object from cache
voter_df.unpersist()
```

In the above code block, `voter_df.cache()` is used to cache the DataFrame, and `voter_df.unpersist()` is used to remove the DataFrame from the cache. The `is_cached` property is used to check if a DataFrame is cached.

Let's see another example:

```python
## Check time for cached DataFrame
start_time = time.time()

# Add caching to the unique rows in departures_df
departures_df = departures_df.distinct().cache()
# Count the unique rows in departures_df, noting how long the operation takes
print("Counting %d rows took %f seconds" % (departures_df.count(), time.time() - start_time))

# Count the rows again, noting the variance in time of a cached DataFrame
start_time = time.time()
print("Counting %d rows again took %f seconds" % (departures_df.count(), time.time() - start_time))

## Takeaway:
# Action only instantiates the caching after `distinct()` function completes!
```

In this example, we first cache the unique rows in `departures_df` using the `distinct()` and `cache()` methods. We then count the rows in the DataFrame and note the time taken. When we count the rows again, the operation is faster because the DataFrame is cached in memory.

Notice that the caching only occurs after the `distinct()` function completes. This is due to Spark's lazy evaluation. The `distinct()` transformation is not computed until the `count()` action is called.

```python
## // Removing DataFrame from cache //

# Determine if departures_df is in the cache
print("Is departures_df cached?: %s" % departures_df.is_cached)
print("Removing departures_df from cache")
# Remove departures_df from the cache
departures_df.unpersist()
# Check the cache status again
print("Is departures_df cached?: %s" % departures_df.is_cached)
```

In this last part, we check if `departures_df` is in the cache using `is_cached`. We then remove it from the cache using `unpersist()` and check the cache status again.

Using caching effectively can significantly improve the performance of your PySpark operations, especially when working with large datasets and performing iterative computations.

### Improving Import Performance

The way you import data into PySpark can significantly impact the performance.

*Spark Clusters* are made of two types of processes:
* **Driver process** - handles task assignments and consolidation of data results from workers
* **Worker process** - handle actual transformation/action of Spark job

Important parameters:
* Number of objects (Files, Network locations, etc)
	* More objects better than larger ones
* Can import via wildcard
```python
airport_df = spark.read.csv('airports-*.txt.gz')
```
* General size of objects
	* Spark performs better if objects are of similar size

**Well-defined schemas** will drastically improve import performance.
* Avoids reading the data multiple times
* Provides validation on import

**How to split objects:**

```
# OS utilities / scripts - get files named chunk-0000 to chunk-9999
split -l 10000 -d largefile chunk-
```

**Write out to Parquet:**

Parquet is a columnar format that is optimized for Spark and can provide significant performance improvements.

```python
df_csv = spark.read.csv('singlelargefile.csv')
df_csv.write.parquet('data.parquet')
df = spark.read.parquet('data.parquet')
```

### Cluster Sizing Tips

Spark contains many configuration settings that can be modified to match needs.

```python
# Reading configuration settings
spark.conf.get(<configuration name>)
# Writing configuration settings
spark.conf.set(<configuration name>)
```

The size of the cluster you are using can also impact performance. A larger cluster can handle more data and perform operations faster, but it is also more expensive. Therefore, it's important to find the right balance based on your data size and budget.

**Cluster Deployment Options:** Single node, Standalone, Managed (YARN, Mesos, Kubernetes)

```python
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
```

Writing Spark configurations example:

```python
# Store the number of partitions in variable
before = departures_df.rdd.getNumPartitions()
# Configure Spark to use 500 partitions (default = 200)
spark.conf.set('spark.sql.shuffle.partitions', 500)

# Recreate the DataFrame using the departures data file
departures_df = spark.read.csv('departures.txt.gz').distinct()
# Print the number of partitions for each instance
print("Partition count before change: %d" % before)
print("Partition count after change: %d" % departures_df.rdd.getNumPartitions())
```

**Tips:**
* Driver node should have double the memory of the worker, fast local storage helpful
* More worker nodes is often better than larger workers, test to find balance. Fast local storage extremely useful.

### Spark Execution Plan

Use `df.explain()` to print (logical and physical) plans to the console for debugging purposes

```python
voter_df = df.select(df['VOTER NAME']).distinct()
voter_df.explain()
```

### Shuffling and Broadcasting

*Shuffling* refers to moving data around the various workers to complete a task.
* Hides overall complexity from user, but can be slow to complete, lowers overall throughput
* Often necessary, but try to minimize.

**To limit shuffling:**
* Limit use of `.repartition(num_partitions)`
	* Use `.coalesce(num_partitions)` instead
* Use care when calling `.join`
* Use `.broadcast()`

**Broadcasting** is a technique used in PySpark to speed up join operations when you are joining a large DataFrame with a small DataFrame. Instead of sending the large DataFrame to all the nodes in the cluster, broadcasting sends the small DataFrame to all nodes, which can result in significant performance improvements.

When you broadcast a DataFrame, Spark keeps a copy of that DataFrame on each worker node, rather than sending it over the network to each node with each task. This can greatly speed up tasks that need to access this DataFrame often. It's especially beneficial when you are performing operations like `join` that need to combine two DataFrames based on a condition.

```python
from pyspark.sql.functions import broadcast
combined_df = df_1.join(broadcast(df_2))
```

In the above code block, we are broadcasting the smaller DataFrame, which can improve the performance of the join operation.

Consider the following example:

```python
# Join the flights_df and aiports_df DataFrames
normal_df = flights_df.join(airports_df, \
flights_df["Destination Airport"] == airports_df["IATA"])

# Show the query plan
normal_df.explain()
```

Remember that table joins in Spark are split between the cluster workers. If the data is not local, various shuffle operations are required and can have a negative impact on performance. Instead, we're going to use Spark's `broadcast` operations to give **each** node a copy of the specified data.

Let's see another example of how to use broadcasting:

```python
# Import the broadcast method from pyspark.sql.functions
from pyspark.sql.functions import broadcast

# Join the flights_df and airports_df DataFrames using broadcasting
broadcast_df = flights_df.join(broadcast(airports_df), \
    flights_df["Destination Airport"] == airports_df["IATA"] )

# Show the query plan and compare against the original
broadcast_df.explain()
```

In this example, we use the `broadcast` function to mark the `airports_df` DataFrame for broadcasting. We then join `flights_df` and `airports_df` on the condition that the "Destination Airport" in `flights_df` is equal to the "IATA" in `airports_df`. The `explain()` function is used to show the query plan, which can help you understand how Spark is executing the operation.

Now, let's compare the performance of a normal join operation and a join operation with broadcasting:

```python
## // Comparison //

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
```

In this block of code, we are comparing the time taken to count the number of rows in a normal DataFrame and a broadcast DataFrame. You can observe that the duration for counting rows in the broadcast DataFrame is less than that of the normal DataFrame, demonstrating the efficiency of broadcasting.

Remember that broadcasting should be used judiciously, as it can consume a lot of memory if the DataFrame being broadcasted is large. Therefore, it's best to use broadcasting when you are dealing with a small DataFrame that can fit into the memory of a single worker node.

**A couple tips in summary:**

- Broadcast the smaller DataFrame. The larger the DataFrame, the more time required to transfer to the worker nodes.
- On small DataFrames, it may be better skip broadcasting and let Spark figure out any optimization on its own.
- If you look at the query execution plan, a broadcastHashJoin indicates you've successfully configured broadcasting.

## Complex Processing and Data Pipelines
***
In data science and big data analytics, data pipelines are sequences of data processing steps where the output of one step is the input to the next. Each step in the pipeline is a unit of work that takes some data as input and produces some data as output. A data pipeline typically comprises of the following components:

- **Input(s):** This is the initial data source(s). It could be in various formats such as CSV, JSON, from web services, databases, and so on.
- **Transformations:** These are operations applied to the data to convert it from one form to another. Examples include `withColumn()`, `.filter()`, `.drop()`, etc.
- **Output(s):** This is the result of the pipeline, and it could be in different forms like CSV, Parquet, database, etc.
- **Validation:** This is the process of checking if the result of the pipeline is correct and reliable.
- **Analysis:** This is the process of examining the result to gain insights or make decisions.

In PySpark, we often build data pipelines to process large datasets. Let's see a quick example:

```python
# Import the data to a DataFrame
departures_df = spark.read.csv('2015-departures.csv.gz', header=True)
# Remove any duration of 0
departures_df = departures_df.filter(departures_df["Actual elapsed time (Minutes)"] > 0)
# Add an ID column
departures_df = departures_df.withColumn('id', F.monotonically_increasing_id())
# Write the file out to JSON format
departures_df.write.json('output.json', mode='overwrite')
```

In this pipeline, we first import the data from a CSV file into a DataFrame. We then filter out rows where the "Actual elapsed time (Minutes)" is 0. After that, we add an 'id' column using the `monotonically_increasing_id()` function. Finally, we write the resulting DataFrame out to a JSON file.

This simple pipeline involves several steps, each transforming the data in some way. Now, let's see some specific data handling techniques often used in PySpark pipelines.

### Data handling techniques

**Removing commented lines:** In many datasets, rows starting with '#' or some other character are used for comments and are not actual data. We can remove these rows using the `filter()` function as shown in the below example:**

```python
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
```

**Removing invalid rows:** Some rows in the dataset might not contain the right number of fields. These can be considered as invalid rows and can be removed using the `filter()` function as shown in the below example:

```python
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
```

**Splitting into columns:** When data in a column is complex, such as being a string that contains multiple fields separated by a delimiter, you may want to split the string and create separate columns. This can be achieved using the `split()` function as shown in the below example:**

```python
# Split the content of _c0 on the tab character (aka, '\t')
split_cols = F.split(annotations_df['_c0'], '\t')

# Add the columns folder, filename, width, and height
split_df = annotations_df.withColumn('folder', split_cols.getItem(0))
split_df = split_df.withColumn('filename', split_cols.getItem(1))
split_df = split_df.withColumn('width', split_cols.getItem(2))
split_df = split_df.withColumn('height', split_cols.getItem(3))

# Add split_cols as a column
split_df = split_df.withColumn('split_cols', split_cols)
```

**Further parsing:** If a column contains a list of complex data, we can use a User-Defined Function (UDF) to parse the data as shown in the below example:**

```python
def retriever(cols, colcount):
  # Return a list of dog data
  return cols[4:colcount]
  
# Define the method as a UDF
udfRetriever = F.udf(retriever, ArrayType(StringType()))
# Create a new column using your UDF
split_df = split_df.withColumn('dog_list', udfRetriever(split_df.split_cols, split_df.colcount))
# Remove the original column, split_cols, and the colcount
split_df = split_df.drop('_c0').drop('split_cols').drop('colcount')
```

### Data Validation

Data validation is an essential step in data pipelines. It involves checking the processed data to ensure it meets the specified requirements and is of good quality. There are many ways to perform data validation. 

**Validate rows via join:**
One common method is to use a join operation to compare the processed data against a set of known valid values:

```python
parsed_df = spark.read.parquet('parsed_data.parquet')
company_df = spark.read.parquet('companies.parquet')
verified_df = parsed_df.join(company_df, parsed_df.company == company_df.company)
```

In this example, the join operation automatically removes any rows in `parsed_df` where the 'company' field does not exist in `company_df`. This ensures that the 'company' field in `verified_df` only contains valid values.

Here's another example:

```python
# Rename the column in valid_folders_df
valid_folders_df = valid_folders_df.withColumnRenamed("_c0","folder")
# Count the number of rows in split_df
split_count = split_df.count()
# Join the DataFrames
joined_df = split_df.join(F.broadcast(valid_folders_df), "folder")
# Compare the number of rows remaining
joined_count = joined_df.count()
print("Before: %d\nAfter: %d" % (split_count, joined_count))
```

In this example, we join `split_df` and `valid_folders_df` on the 'folder' column. Any rows in `split_df` where the 'folder' field does not exist in `valid_folders_df` are automatically removed. This ensures that the 'folder' field in `joined_df` only contains valid values.

**Examining invalid rows:**
You can also examine the invalid rows using a 'left_anti' join:

```python
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
```

In this example, we create a DataFrame `invalid_df` that contains the rows in `split_df` where the 'filename' field does not exist in `joined_df`. This gives us the rows that were removed in the join operation. We can then examine these rows to understand why they were removed.

### Final Analysis and Delivery

Once we have cleaned and prepared our data, we can perform some final analysis and computations. PySpark provides several methods for this, including User-Defined Functions (UDFs) and inline calculations.

**Calculations using UDF:**

User-Defined Functions (UDFs) allow you to extend the functionality of PySpark by defining your custom functions to apply on Spark DataFrames. Here's an example:

```python
def getAvgSale(saleslist):
    totalsales = 0
    count = 0
    for sale in saleslist:
        totalsales += sale[2] + sale[3]
        count += 2
    return totalsales / count


udfGetAvgSale = udf(getAvgSale, DoubleType())
df = df.withColumn('avg_sale', udfGetAvgSale(df.sales_list))
```

In this example, we first define a Python function `getAvgSale` that calculates the average sale from a list of sales. We then convert this function into a UDF using `F.udf`. Finally, we use this UDF in `withColumn` to apply the function on the 'sales_list' column and create a new column 'avg_sale'.

**Inline calculations:**

In addition to using UDFs, we can also perform calculations directly on DataFrame columns:

```python
df = spark.read.csv('datafile')
df = df.withColumn('avg', (df['total_sales'] / df['sales_count'])) # Calculate the 'avg' column
df = df.withColumn('sq_ft', df['width'] * df['length']) # Calculate the 'sq_ft' column
df = df.withColumn('total_avg_size', udfComputeTotal(df.entries) / df.numEntries) # Using UDF
```

In this example, we perform some simple arithmetic operations on the DataFrame columns to calculate averages and areas.

***

### Dog Parsing

In this section, we will demonstrate how to parse and analyze complex data in a DataFrame. We will use a dataset of dog images, where each row contains a list of details about the dogs in an image.

```python
# Select the dog details and show 10 untruncated rows
print(joined_df.select("dog_list").show(10, truncate=False))

# Example of Output:
# |[affenpinscher,0,9,173,298] | 
# |[Border_terrier,73,127,341,335] |
```

In this dataset, the 'dog_list' column contains a list of details about each dog in an image. Each detail is a string that includes the dog's breed and bounding box coordinates (start_x, start_y, end_x, end_y) in the image.

To make this data easier to work with, we will define a schema for the dog details:

```python
# Define a schema type for the details in the dog list
DogType = StructType([
	StructField("breed", StringType(), False),
    StructField("start_x", IntegerType(), False),
    StructField("start_y", IntegerType(), False),
    StructField("end_x", IntegerType(), False),
    StructField("end_y", IntegerType(), False)
])
```

This schema represents the structure of the dog details. It can be used to convert the string of dog details into a more structured format.

Next, we will create a function to parse the string of dog details and convert it into a list of structured data:

```python
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
```

In this block of code, we define a function `dogParse` that parses the string of dog details into a list of structured data according to the `DogType` schema. We then convert this function into a UDF and apply it to the 'dog_list' column to create a new 'dogs' column. The old 'dog_list' column is dropped.

Next, we can use another UDF to calculate the number of pixels in an image that represent dogs:

```python
# Define a UDF to determine the number of pixels per image
def dogPixelCount(doglist):
  totalpixels = 0
  for dog in doglist:
	  # Compute bounding box using: (Xend - Xstart) * (Yend - Ystart)
    totalpixels += (dog[3] - dog[1]) * (dog[4] - dog[2])
  return totalpixels

# Define a UDF for the pixel count
udfDogPixelCount = F.udf(dogPixelCount, IntegerType())
joined_df = joined_df.withColumn('dog_pixels', udfDogPixelCount('dogs'))
```

In this block of code, we define a function `dogPixelCount` that calculates the total number of pixels that represent dogs in an image. We then convert this function into a UDF and apply it to the 'dogs' column to create a new 'dog_pixels' column.

Finally, we can calculate the percentage of pixels in an image that represent dogs and display images where this percentage is more than 60%:

```python
# Create a column representing the percentage of pixels
# Total "dog pixels" / Total size of image
joined_df = joined_df.withColumn('dog_percent', (joined_df.dog_pixels / (joined_df.width * joined_df.height)) * 100)

# Show the first 10 annotations with more than 60% dog
joined_df.where('dog_percent > 60').show(10)
```

In this block of code, we first calculate the percentage of pixels that represent dogs in each image by dividing the 'dog_pixels' column by the total size of the image (width * height). We then display the first 10 images where this percentage is more than 60%.

## Conclusion
***
In this guide, we covered a variety of techniques and functionalities available in PySpark for data cleaning and transformation. We discussed basic and advanced data selection and filtering, column transformations, ArrayType column functions, conditional DataFrame operations, User-Defined Functions (UDFs), partitioning and lazy processing, performance improvement techniques, handling complex data pipelines, and data validation. With these techniques in hand, you can tackle a wide range of data cleaning and transformation tasks in PySpark.
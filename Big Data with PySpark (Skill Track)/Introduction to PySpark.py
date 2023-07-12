## =================================
## COURSE 1: INTRODUCTION TO PYSPARK
## =================================

from pyspark.sql import SparkSession

## Create new or retrieve existing Spark Session

spark = SparkSession.builder.getOrCreate()
print(spark)

## Print the tables in the catalog

print(spark.catalog.listTables())

## Running SQL queries on tables in Spark cluster
# Example: Obtain first 10 rows of flights

flights10 = spark.sql("FROM flights SELECT * LIMIT 10")
flights10.show()

## Converting Query to Pandas DataFrame

flight_counts = spark.sql("SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest")
pd_counts = flight_counts.toPandas()
print(pd_counts.head())

## Insert Pandas DataFrame into Spark cluster

pd_temp = pd.DataFrame(np.random.random(10))
spark_temp = spark.createDataFrame(pd_temp) # create Spark DataFrame
print(spark.catalog.listTables())
spark_temp.createOrReplaceTempView() # Register `spark_temp` DataFrame
print(spark.catalog.listTables())

## Reading text file direct into Spark cluster

file_path = "/usr/local/share/datasets/airports.csv"
# Read in the airports data
airports = spark.read.csv(file_path, header = True)
airports.show() # Show the data

## --------------------------------------------------------------- ##
## Manipulating Data ##

## Creating columns

flights = spark.table("flights") # Create Spark DataFrame
flights.show() # Display head
flights = flights.withColumn("duration_hrs", flights.air_time / 60)

## Filtering Data

# Filter flights by passing a string vs passing a column of boolean values
long_flights1 = flights.filter("distance > 1000")
long_flights2 = flights.filter(flights.distance > 1000)

# Print the data to check they're equal (i.e. operations are equivalent)
long_flights1.show()
long_flights2.show()

## Selecting Data

selected1 = flights.select("tailnum", "origin", "dest")
temp = flights.select(flights.origin, flights.dest, flights.carrier)

filterA = flights.origin == "SEA"
filterB = flights.dest == "PDX"

selected2 = temp.filter(filterA).filter(filterB)

"""
The difference between .select() and .withColumn() methods is 
that .select() returns only the columns you specify, 
while .withColumn() returns all the columns of the DataFrame 
in addition to the one you defined. 
"""

## Performing column-wise operations with Select

# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")

# Select columns
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

# Create the same table using a SQL expression
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")

## Aggregating

# Find the shortest flight from PDX in terms of distance
flights.filter(flights.origin == "PDX").groupBy().min("distance").show()

# Find the longest flight from SEA in terms of air time
flights.filter(flights.origin == "SEA").groupBy().max("air_time").show()

## Aggregating II

# Average duration of Delta flights
flights.filter(flights.carrier == "DL").filter(flights.origin == "SEA").groupBy().avg("air_time").show()

# Total hours in the air
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum("duration_hrs").show()

## Grouping and Aggregating I

# Count number of flights each plane made
by_plane = flights.groupBy("tailnum")
by_plane.count().show()

# Average duration of flights from PDX and SEA
by_origin = flights.groupBy("origin")
by_origin.avg("air_time").show()

## Grouping and Aggregating II

# Import pyspark.sql.functions as F
import pyspark.sql.functions as F

# Group by month and dest
by_month_dest = flights.groupBy("month", "dest")

# Average departure delay by month and destination
by_month_dest.avg("dep_delay").show()

# Standard deviation of departure delay
by_month_dest.agg(F.stddev("dep_delay")).show()

## Joins

# Examine the data
print(airports.show())

# Rename the faa column
airports = airports.withColumnRenamed("faa", "dest")

# Join the DataFrames
flights_with_airports = flights.join(airports, 'dest', 'leftouter')

# Examine the new DataFrame
print(flights_with_airports.show())

## --------------------------------------------------------------- ##
## Getting started with Machine Learning Pipelines ##

## Join the DataFrames

# Rename year column
planes = planes.withColumnRenamed("year", "plane_year")
# Join the DataFrames
model_data = flights.join(planes, on="tailnum", how="leftouter")

## String to integer

# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast("integer"))
model_data = model_data.withColumn("month", model_data.month.cast("integer"))
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast("integer"))

# Create the column plane_age
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)

# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn("label", model_data.is_late.cast("integer"))

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")

# Create a StringIndexer
carr_indexer = StringIndexer(inputCol = "carrier", outputCol = "carrier_index")

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol = "carrier_index", outputCol = "carrier_fact")

# Create a StringIndexer
dest_indexer = StringIndexer(inputCol = "dest", outputCol = "dest_index")

# Create a OneHotEncoder
dest_encoder = OneHotEncoder(inputCol = "dest_index", outputCol = "dest_fact")

# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")

# Create ML Pipeline
from pyspark.ml import Pipeline
flights_pipe = Pipeline(stages = [dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])

# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)

# Split the data into training and test sets
training, test = piped_data.randomSplit([.6, .4])

# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression Estimator
lr = LogisticRegression()

# Import the evaluation submodule
import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName = "areaUnderROC")

# Hyperparameter grid search
# Import the tuning submodule
import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the grid
grid = grid.build()

# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )

# Call lr.fit()
best_lr = lr.fit(training)

# Print best_lr
print(best_lr)

# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))

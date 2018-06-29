import os
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages=com.amazonaws:aws-java-sdk-pom:1.10.34,org.apache.hadoop:hadoop-aws:2.7.3 pyspark-shell"
import pyspark
from datetime import datetime
# import seaborn as sns
import sys
# import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np

print('Python version ' + sys.version)
print('Spark version: ' + pyspark.__version__)

import string
import random

# Set the configuration
# random string for instance name
inst = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
AppName = inst + ' - Ceph S3 Prometheus JSON Reader'
conf = pyspark.SparkConf().setAppName(AppName).setMaster('local[2]')
print("Application Name: ", AppName)

# specify number of nodes need (1-5)
conf.set("spark.cores.max", "8")

# specify Spark executor memory (default is 1gB)
conf.set("spark.executor.memory", "4g")

# Set the Spark cluster connection
sc = pyspark.SparkContext.getOrCreate(conf)

# Set the Hadoop configurations to access Ceph S3
# import os
(ceph_key, ceph_secret, ceph_host) = (os.getenv('DH_CEPH_KEY'), os.getenv('DH_CEPH_SECRET'), os.getenv('DH_CEPH_HOST'))

sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", ceph_key)
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", ceph_secret)
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", ceph_host)

#Get the SQL context
sqlContext = pyspark.SQLContext(sc)

# specify metric we want to analyze
metric_name = 'kubelet_docker_operations_latency_microseconds'

# choose a label
label = ""

# specify for histogram metric type
bucket_val = '0.3'

# specify for summary metric type
quantile_val = '0.99'

# specify any filtering when collected the data
# For example:
# If I want to just see data from a specific host, specify "metric.hostname='free-stg-master-03fb6'"
where_labels = ["metric.hostname='free-stg-master-03fb6'"]

jsonUrl = "s3a://DH-DEV-PROMETHEUS-BACKUP/prometheus-openshift-devops-monitor.1b7d.free-stg.openshiftapps.com/" + metric_name

try:
    jsonFile_sum = sqlContext.read.option("multiline", True).option("mode", "PERMISSIVE").json(jsonUrl + '_sum/')
    jsonFile = sqlContext.read.option("multiline", True).option("mode", "PERMISSIVE").json(jsonUrl + '_count/')
    try:
        jsonFile_bucket = sqlContext.read.option("multiline", True).option("mode", "PERMISSIVE").json(jsonUrl + '_bucket/')
        metric_type = 'histogram'
    except:
        jsonFile_quantile = sqlContext.read.option("multiline", True).option("mode", "PERMISSIVE").json(jsonUrl+'/')
        metric_type = 'summary'
except:
    jsonFile = sqlContext.read.option("multiline", True).option("mode", "PERMISSIVE").json(jsonUrl+'/')
    metric_type = 'gauge or counter'

#Display the schema of the file
print("Metric Type: ", metric_type)
print("Schema:")
jsonFile.printSchema()

labels = []
for i in jsonFile.schema["metric"].jsonValue()["type"]["fields"]:
    labels.append(i["name"])

print("number of labels: ", len(labels))
print("\n===== Labels =====")
inc = 0
for i in labels:
    inc = inc+1
    print(inc, "\t", i)

prompt = "\n===== Select a Label (specify number from 0 to " + str(len(labels)) + "\n0 indicates no label to select\n"
# label_num = int(input(prompt))
label_num = 10
if label_num != 0:
    label = labels[label_num - 1]
print("\n===== Label Selected: ",label)

import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType

# create function to convert POSIX timestamp to local date
def convert_timestamp(t):
    return str(datetime.fromtimestamp(int(t)))

def format_df(df):
    #reformat data by timestamp and values
    df = df.withColumn("values", F.explode(df.values))

    df = df.withColumn("timestamp", F.col("values").getItem(0))
    df = df.sort("timestamp", ascending=True)

    df = df.withColumn("values", F.col("values").getItem(1))

    # drop null values
    df = df.na.drop(subset=["values"])

    # cast values to int
    df = df.withColumn("values", df.values.cast("int"))

    # define function to be applied to DF column
    udf_convert_timestamp = F.udf(lambda z: convert_timestamp(z), StringType())

    # convert timestamp values to datetime timestamp
    df = df.withColumn("timestamp", udf_convert_timestamp("timestamp"))

    # calculate log(values) for each row
    df = df.withColumn("log_values", F.log(df.values))

    return df

def extract_from_json(json, name, select_labels, where_labels):
    #Register the created SchemaRDD as a temporary variable
    json.registerTempTable(name)

    #Filter the results into a data frame

    query = "SELECT values"

    # check if select labels are specified and add query condition if appropriate
    if len(select_labels) > 0:
        query = query + ", " + ", ".join(select_labels)

    query = query + " FROM " + name

    # check if where labels are specified and add query condition if appropriate
    if len(where_labels) > 0:
        query = query + " WHERE " + " AND ".join(where_labels)

    print(query)
    data = sqlContext.sql(query)

    # sample data to make it more manageable
    # data = data.sample(False, fraction = 0.05, seed = 0)
    # TODO: get rid of this hack
    data = sqlContext.createDataFrame(data.head(1000), data.schema)

    return format_df(data)

if label != "":
    select_labels = ['metric.' + label]
else:
    select_labels = []

# get data and format
data = extract_from_json(jsonFile, metric_name, select_labels, where_labels)

data.count()
data.show()

def calculate_sample_rate(df):
    # define function to be applied to DF column
    udf_timestamp_hour = F.udf(lambda dt: int(datetime.strptime(dt,'%Y-%m-%d %X').hour), IntegerType())

    # convert timestamp values to datetime timestamp

    # new df with hourly value count
    vals_per_hour = df.withColumn("hour", udf_timestamp_hour("timestamp")).groupBy("hour").count()

    # average density (samples/hour)
    avg = vals_per_hour.agg(F.avg(F.col("count"))).head()[0]
    print("average hourly sample count: ", avg)

    # sort and display hourly count
    vals_per_hour.sort("hour").show()

calculate_sample_rate(data)

def calculate_vals_per_label(df):
    # new df with vals per label
    df.groupBy(label).count().show()

calculate_vals_per_label(data)

from pyspark.sql.window import Window

def get_deltas(df):
    df_lag = df.withColumn('prev_vals',
                        F.lag(df['values'])
                                 .over(Window.partitionBy("timestamp").orderBy("timestamp")))

    result = df_lag.withColumn('deltas', (df_lag['values'] - df_lag['prev_vals']))
    result = result.drop("prev_vals")

    max_delta = result.agg(F.max(F.col("deltas"))).head()[0]
    min_delta = result.agg(F.min(F.col("deltas"))).head()[0]
    mean_delta = result.agg(F.avg(F.col("deltas"))).head()[0]

    return max_delta, min_delta, mean_delta, result

max_delta, min_delta, mean_delta, data = get_deltas(data)
data.show()
print("Max delta:", max_delta)
print("Min delta:", min_delta)
print("Mean delta:", mean_delta)

# rhmax = current value / max(values)
# if rhmax (of new point) >> 1, we can assume that the point is an anomaly
'''
def get_rhmax(df):
    real_max = df.agg(F.max(F.col("values"))).head()[0]
    result = df.withColumn("rhmax", df["values"]/real_max)
    return result

result = get_rhmax(data)
result.show()

def gauge_counter_separator(df):
    vals = np.array(data.select("values").collect())
    diff = vals - np.roll(vals, 1) # this value - previous value (should always be zero or positive for counter)
    diff[0] = 0 # ignore first difference, there is no value before the first
    diff[np.where(vals == 0)] = 0
    # check if these are any negative differences, if not then metric is a counter.
    # if counter, we must convert it to a gauge by keeping the derivatives
    if ((diff < 0).sum() == 0):
        metric_type = 'counter'
    else:
        metric_type = 'gauge'
    return metric_type, df

if metric_type == "gauge or counter":
    metric_type, data = gauge_counter_separator(data)

if metric_type == 'histogram':
    data_sum = extract_from_json(jsonFile_sum, metric_name, select_labels, where_labels)

    select_labels.append("metric.le")
    data_bucket = extract_from_json(jsonFile_bucket, metric_name, select_labels, where_labels)

    # filter by specific le value
    data_bucket = data_bucket.filter(bucket_val)

    data_sum.show()
    data_bucket.show()

elif metric_type == 'summary':
    # get metric sum data
    data_sum = extract_from_json(jsonFile_sum, metric_name, select_labels, where_labels)

    # get metric quantile data
    select_labels.append("metric.quantile")
    data_quantile = extract_from_json(jsonFile_quantile, metric_name, select_labels, where_labels)

    # filter by specific quantile value
    data_quantile = data_quantile.filter(data_quantile.quantile == quantile_val)
    # get rid of NaN values once again. This is required once filtering takes place
    data_quantile = data_quantile.na.drop(subset='values')

    data_sum.show()
    data_quantile.show()

def get_stats(df):
    # calculate mean
    mean = df.agg(F.avg(F.col("values"))).head()[0]

    # calculate variance
    var = df.agg(F.variance(F.col("values"))).head()[0]

    # calculate standard deviation
    stddev = df.agg(F.stddev(F.col("values"))).head()[0]

    # calculate median
    median = float(df.approxQuantile("values", [0.5], 0.25)[0])

    return mean, var, stddev, median

mean, var, stddev, median = get_stats(data)

print("\tMean(values): ", mean)
print("\tVariance(values): ", var)
print("\tStddev(values): ", stddev)
print("\tMedian(values): ", median)

sns.set(color_codes = True)

def plot_hist(df, axis_label):
    vals = np.array(df.select("values").collect())

    if np.count_nonzero(vals) == 0:
        return "Error: All values are zero"

    # log normalization for postiive numbers only
    log_vals = np.log(vals[vals != 0])

    # plot both the distribution and the normalized distribution
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,12))
    sns.distplot(vals, kde=True, ax=ax[0], axlabel=axis_label)
    sns.distplot(log_vals, kde=True, ax=ax[1], axlabel = "log transformed "+ axis_label)
    fig.show()

plot_hist(data, metric_name)

# df is the Spark dataframe
# filter label is x-axis, ie. the metric label which we want to categorize the data by
def plot_box_whisker(df, filter_label):
    plt.figure(figsize=(20,15))
    df = df.withColumn("log_values", F.log(df.log_values))
    ax = sns.boxplot(x=filter_label, y="log_values", hue=filter_label, data=df.toPandas())  # RUN PLOT

    plt.show()
    plt.clf()
    plt.close()

if label != "":
    plot_box_whisker(data, label)

#TODO: downsampling
data = data.sort("timestamp", ascending=True)
vals = np.array(data.select("values").collect())
plt.plot(vals)
plt.xlabel("timestamp")
plt.ylabel("value")
'''
# be sure to stop the Spark Session to conserve resources
sc.stop()

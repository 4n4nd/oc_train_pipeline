import pandas as pd
import json
import numpy as np
import sys
print("Importing pyspark....")
import pyspark
print("....done")
import os
from fbprophet import Prophet
import random
import string
from datetime import datetime
from datetime import timedelta

# from lib.ceph import CephConnect as cp

import warnings
warnings.filterwarnings('ignore')

SPARK_MASTER="spark://" + os.getenv('OSHINKO_CLUSTER_NAME','spark-cluster.dh-prod-analytics-factory.svc') + ":7077"
metric_name = os.getenv('PROM_METRIC_NAME','kubelet_docker_operations_latency_microseconds')
label = os.getenv('LABEL',"operation_type")
# start_time = os.getenv('BEGIN_TIMESTAMP')
# end_time = os.getenv('END_TIMESTAMP')

# SPARK_MASTER = 'spark://spark-cluster.dh-prod-analytics-factory.svc:7077'
# metric_name = 'kubelet_docker_operations_latency_microseconds'
start_time = datetime(2018, 6, 1)
end_time = datetime(2018, 6, 26)
# label = "operation_type"
OP_TYPE = 'list_images'

bucket_val = '0.3'
quantile_val = '0.9'
where_labels = {}


# Set the configuration
# random string for instance name
inst = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
AppName = inst + ' - Ceph S3 Prophet Forecasting'
#Set the configuration
conf = pyspark.SparkConf().setAppName(AppName).setMaster(SPARK_MASTER)
print("Application Name: ", AppName)

# specify number of nodes need (1-5)
conf.set("spark.cores.max", "8")

# specify Spark executor memory (default is 1gB)
conf.set("spark.executor.memory", "10g")

#Set the Spark cluster connection
sc = pyspark.SparkContext.getOrCreate(conf)

#Set the Hadoop configurations to access Ceph S3
sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", os.getenv('DH_CEPH_KEY'))
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", os.getenv('DH_CEPH_SECRET'))
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", os.getenv('DH_CEPH_HOST'))


#Get the SQL context
sqlContext = pyspark.SQLContext(sc)


#Read the Prometheus JSON BZip data
# jsonFile = sqlContext.read.option("multiline", True).option("mode", "PERMISSIVE").json("s3a://DH-DEV-PROMETHEUS-BACKUP/prometheus-openshift-devops-monitor.1b7d.free-stg.openshiftapps.com/"+metric_name+"/")
prom_host = "prometheus-openshift-devops-monitor.1b7d.free-stg.openshiftapps.com"
jsonUrl = "s3a://DH-DEV-PROMETHEUS-BACKUP/prometheus-openshift-devops-monitor.1b7d.free-stg.openshiftapps.com/" + metric_name
# jsonUrl = "s3a://" + os.getenv('DH_CEPH_BUCKET','DH-DEV-PROMETHEUS-BACKUP') + "/" + prom_host + "/" + metric_name
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
#print("Metric Type: ", metric_type)
print("Schema:")
jsonFile.printSchema()


import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import TimestampType

# create function to convert POSIX timestamp to local date
def convert_timestamp(t):
    return datetime.fromtimestamp(int(t))

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
    udf_convert_timestamp = F.udf(lambda z: convert_timestamp(z), TimestampType())

    # convert timestamp values to datetime timestamp
    df = df.withColumn("timestamp", udf_convert_timestamp("timestamp"))

    # calculate log(values) for each row
    df = df.withColumn("log_values", F.log(df.values))

    df = df.na.drop(subset=["values"])

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

    #sample data to make it more manageable
    #data = data.sample(False, fraction = 0.05, seed = 0)
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

def in_time_frame(val, start, end):
    if val >= start and val <= end:
        return 1
    return 0

def get_data_in_timeframe(df, start_time, end_time):
    udf_in_time_frame = F.udf(lambda z: in_time_frame(z, start_time, end_time), StringType())

    # convert timestamp values to datetime timestamp
    df = df.withColumn("check_time", udf_in_time_frame("timestamp"))
    df = df.withColumn("check_time", F.log(df.check_time))
    df = df.na.drop(subset = ["check_time"])
    df = df.drop("check_time")
    return df

data = get_data_in_timeframe(data, start_time, end_time)
data.show()

def calculate_sample_rate(df):
    # define function to be applied to DF column
    udf_timestamp_hour = F.udf(lambda dt: dt.replace(minute=0, second=0) + timedelta(hours=1), TimestampType())


    # convert timestamp values to datetime timestamp

    # new df with hourly value count
    vals_per_hour = df.withColumn("hourly", udf_timestamp_hour("timestamp")).groupBy("hourly").count()

    # average density (samples/hour)
    avg = vals_per_hour.agg(F.avg(F.col("count"))).head()[0]
    print("average hourly sample count: ", avg)

    # sort and display hourly count
    vals_per_hour.sort("hourly").show()

calculate_sample_rate(data)

def calculate_vals_per_label(df, df_label):
    # new df with vals per label
    df.groupBy(df_label).count().show()

calculate_vals_per_label(data, label)

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
def get_rhmax(df):
    real_max = df.agg(F.max(F.col("values"))).head()[0]
    result = df.withColumn("rhmax", df["values"]/real_max)
    return result

result = get_rhmax(data)
result.show()

def gauge_counter_separator(df):
    vals = np.array(df.select("values").collect())
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

print("Metric type: ", metric_type)


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

data_pd = data.toPandas()
del data
# be sure to stop the Spark Session to conserve resources
sc.stop()

from fbprophet import Prophet

#temp_frame = get_filtered_op_frame(OP_TYPE)
data_pd = data_pd.set_index(data_pd.timestamp)
data_pd = data_pd[['timestamp','values']]
OP_TYPE = 'list_images'
data_pd = data_pd.filter(data_pd.operation_type == OP_TYPE)

train_frame = data_pd[0 : int(0.7*len(data_pd))]
test_frame = data_pd[int(0.7*len(data_pd)) : ]

print(len(train_frame), len(test_frame), len(data_pd))

train_frame['y'] = train_frame['values']
train_frame['ds'] = train_frame['timestamp']


m = Prophet()

m.fit(train_frame)

future = m.make_future_dataframe(periods= int(len(test_frame) * 1.1),freq= '1MIN')
forecast = m.predict(future)
forecast.head()

forecasted_features = ['ds','yhat','yhat_lower','yhat_upper']
# m.plot(forecast,xlabel="Time stamp",ylabel="Value");
# m.plot_components(forecast);

forecast = forecast[forecasted_features]
forecast.head()

import pandas as pd
forecast['timestamp'] = forecast['ds']
forecast = forecast.set_index(forecast.timestamp)

# Store prediction to ceph
session = cp()
object_path = "Predictions" + "/" + prom_host + "/" + metric_name + "_" + str(start_time) + "_" + end_time + ".json"
session.store_data(name = metric_name, object_path = object_path, values = forecast.to_json())


#forecast.head()
test_frame['timestamp'] = pd.to_datetime(test_frame.timestamp)
#test_frame.head()
joined_frame = pd.merge(test_frame, forecast, how='inner', on=['timestamp'],left_index=True)

joined_frame.head()
joined_frame.count()

joined_frame['residuals'] = joined_frame['values'] - joined_frame['yhat']

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
# sns.distplot(joined_frame['residuals'], kde=True, axlabel= 'residuals',bins=50)
# fig.show()

mean_residuals = np.mean(joined_frame['residuals'])
std_residuals = np.std(joined_frame['residuals'])
one_std = mean_residuals + std_residuals
two_std = mean_residuals + 2 * std_residuals
three_std = mean_residuals + 3 * std_residuals
print("Mean:",mean_residuals," STD:",std_residuals,"1_STD:",one_std,"2_STD:",two_std,"3_STD:",three_std)

value_count_1_STD = len(joined_frame[(joined_frame['residuals']<= mean_residuals + one_std) & (joined_frame['residuals']>= mean_residuals - one_std)])
value_count_2_STD = len(joined_frame[(joined_frame['residuals']<= mean_residuals + two_std) & (joined_frame['residuals']>= mean_residuals - two_std)])
value_count_3_STD = len(joined_frame[(joined_frame['residuals']<= mean_residuals + three_std) & (joined_frame['residuals']>= mean_residuals - three_std)])

print(" Mean + 1_STD is: ", int(value_count_1_STD/len(joined_frame) * 100), "% of total values")
print(" Mean + 2_STD is: ", int(value_count_2_STD/len(joined_frame) * 100), "% of total values")
print(" Mean + 3_STD is: ", int(value_count_3_STD/len(joined_frame) * 100), "% of total values")

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
# sns.distplot((joined_frame['residuals'] - mean_residuals)/one_std, kde=True, axlabel= 'residuals',bins=50)
# fig.show()

joined_frame.set_index(joined_frame['timestamp'],inplace= True)
joined_frame2 = joined_frame[:int(0.3 * len(joined_frame))]
# joined_frame2[['values','yhat','yhat_lower','yhat_upper']].plot(figsize=(15,12));
test_frame['timestamp'].max()
#Register the created SchemaRDD as a temporary table.

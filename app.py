import pandas as pd
import json
import numpy as np
import sys
import os
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages=org.apache.hadoop:hadoop-aws:2.7.3 pyspark-shell"
import pyspark
from fbprophet import Prophet
import random
import string
import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from lib.ceph import CephConnect as cp
from lib.spark import *
metric_name = os.getenv('PROM_METRIC_NAME','kubelet_docker_operations_latency_microseconds')
METRIC_NAME = metric_name
label = os.getenv('LABEL',"operation_type")
LABEL = label
# start_time = os.getenv('BEGIN_TIMESTAMP')
# end_time = os.getenv('END_TIMESTAMP')

# SPARK_MASTER = 'spark://spark-cluster.dh-prod-analytics-factory.svc:7077'
# metric_name = 'kubelet_docker_operations_latency_microseconds'
start_time = 1530973037
end_time = 1531837110

START_TIME = start_time
END_TIME = end_time
# label = "operation_type"
OP_TYPE = 'create'

bucket_val = '0.3'
quantile_val = '0.99'
where_labels = ["metric.hostname='free-stg-master-03fb6'"]


spark_connect = SparkConnect(spark_cores=12,spark_memory="14g")
sqlContext = spark_connect.get_sql_context()

print("Data start time", datetime.datetime.fromtimestamp(START_TIME))
print("Data end time", datetime.datetime.fromtimestamp(END_TIME))

#Read the Prometheus JSON BZip data
# jsonFile = sqlContext.read.option("multiline", True).option("mode", "PERMISSIVE").json("s3a://DH-DEV-PROMETHEUS-BACKUP/prometheus-openshift-devops-monitor.1b7d.free-stg.openshiftapps.com/"+metric_name+"/")
prom_host = "prometheus-openshift-devops-monitor.1b7d.free-stg.openshiftapps.com"

jsonUrl = "s3a://DH-DEV-PROMETHEUS-BACKUP/prometheus-openshift-devops-monitor.1b7d.free-stg.openshiftapps.com/" + METRIC_NAME
print(jsonUrl+'/')

jsonFile = sqlContext.read.option("multiline", True).option("mode", "PERMISSIVE").json(jsonUrl+'/')

print("Schema:")
jsonFile.printSchema()


import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import TimestampType

# create function to convert POSIX timestamp to local date
def convert_timestamp(t):
    return datetime.datetime.fromtimestamp(float(t))

def format_df(df):
    #reformat data by timestamp and values
    df = df.withColumn("values", F.explode(df.values))

    df = df.withColumn("timestamp", F.col("values").getItem(0))

    df = df.withColumn("values", F.col("values").getItem(1))

    # drop null values
    df = df.na.drop(subset=["values"])

    # cast values to int
    df = df.withColumn("values", df.values.cast("int"))

    # define function to be applied to DF column
    udf_convert_timestamp = F.udf(lambda z: convert_timestamp(z), TimestampType())

    # convert timestamp values to datetime timestamp
    df = df.withColumn("timestamp", udf_convert_timestamp("timestamp"))

    # drop null values
    #df = df.na.drop(subset=["values"])

    # calculate log(values) for each row
    #df = df.withColumn("log_values", F.log(df.values))

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

    print("SQL QUERRY: ", query)
    df = sqlContext.sql(query)

    #sample data to make it more manageable
    #data = data.sample(False, fraction = 0.05, seed = 0)
    # TODO: get rid of this hack
    #df = sqlContext.createDataFrame(df.head(1000), df.schema)

    return format_df(df)

if LABEL != "":
    select_labels = ['metric.' + LABEL]
else:
    select_labels = []

# get data and format
data = extract_from_json(jsonFile, METRIC_NAME, select_labels, where_labels)

data.count()
data.show()





data = data.filter(F.col("timestamp") > datetime.datetime.fromtimestamp(START_TIME))
data = data.filter(F.col("timestamp") < datetime.datetime.fromtimestamp(END_TIME))
data.count()



data_pd = data.toPandas()
del data
# be sure to stop the Spark Session to conserve resources
spark_connect.stop_spark()

# Delete Spark Cluster


from fbprophet import Prophet

#temp_frame = get_filtered_op_frame(OP_TYPE)
OP_TYPE = 'sync'
data_pd = data_pd[data_pd['operation_type'] == OP_TYPE]


train_frame = data_pd[0 : int(0.7*len(data_pd))]
test_frame = data_pd[int(0.7*len(data_pd)) : ]

print(len(train_frame), len(test_frame), len(data_pd))

train_frame['y'] = train_frame['values']
train_frame['ds'] = train_frame['timestamp']


m = Prophet()

m.fit(train_frame)

future = m.make_future_dataframe(periods= int(len(test_frame) * 1.1),freq= '1MIN')
forecast = m.predict(future)
print(forecast.head())

forecasted_features = ['ds','yhat','yhat_lower','yhat_upper']
# m.plot(forecast,xlabel="Time stamp",ylabel="Value");
# m.plot_components(forecast);

forecast = forecast[forecasted_features]
print(forecast.head())

forecast['timestamp'] = forecast['ds']
forecast['values'] = test_frame['values']
forecast = forecast[['timestamp','values','yhat','yhat_lower','yhat_upper']]

# Store Forecast to CEPH
start_time = datetime.datetime.fromtimestamp(start_time)
end_time = datetime.datetime.fromtimestamp(end_time)

session = cp()
object_path = "Predictions" + "/" + prom_host + "/" + metric_name + "_" + (start_time.strftime("%Y%m%d%H%M")) + "_" + (end_time.strftime("%Y%m%d%H%M")) + ".json"
print(session.store_data(name = metric_name, object_path = object_path, values = forecast.to_json()))


# import pandas as pd
#
# forecast = forecast.set_index(forecast.timestamp)
# #forecast.head()
#
#
#
# test_frame['timestamp'] = pd.to_datetime(test_frame.timestamp)
# #test_frame.head()
# joined_frame = pd.merge(test_frame, forecast, how='inner', on=['timestamp'],left_index=True)
#
# joined_frame.head()
# joined_frame.count()
#
# joined_frame['residuals'] = joined_frame['values'] - joined_frame['yhat']
#
# # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
# # sns.distplot(joined_frame['residuals'], kde=True, axlabel= 'residuals',bins=50)
# # fig.show()
#
# mean_residuals = np.mean(joined_frame['residuals'])
# std_residuals = np.std(joined_frame['residuals'])
# one_std = mean_residuals + std_residuals
# two_std = mean_residuals + 2 * std_residuals
# three_std = mean_residuals + 3 * std_residuals
# print("Mean:",mean_residuals," STD:",std_residuals,"1_STD:",one_std,"2_STD:",two_std,"3_STD:",three_std)
#
# value_count_1_STD = len(joined_frame[(joined_frame['residuals']<= mean_residuals + one_std) & (joined_frame['residuals']>= mean_residuals - one_std)])
# value_count_2_STD = len(joined_frame[(joined_frame['residuals']<= mean_residuals + two_std) & (joined_frame['residuals']>= mean_residuals - two_std)])
# value_count_3_STD = len(joined_frame[(joined_frame['residuals']<= mean_residuals + three_std) & (joined_frame['residuals']>= mean_residuals - three_std)])
#
# print(" Mean + 1_STD is: ", int(value_count_1_STD/len(joined_frame) * 100), "% of total values")
# print(" Mean + 2_STD is: ", int(value_count_2_STD/len(joined_frame) * 100), "% of total values")
# print(" Mean + 3_STD is: ", int(value_count_3_STD/len(joined_frame) * 100), "% of total values")
#
# # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
# # sns.distplot((joined_frame['residuals'] - mean_residuals)/one_std, kde=True, axlabel= 'residuals',bins=50)
# # fig.show()
#
# joined_frame.set_index(joined_frame['timestamp'],inplace= True)
# joined_frame2 = joined_frame[:int(0.3 * len(joined_frame))]
# # joined_frame2[['values','yhat','yhat_lower','yhat_upper']].plot(figsize=(15,12));
# test_frame['timestamp'].max()
# #Register the created SchemaRDD as a temporary table.

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
from datetime import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from lib.ceph import CephConnect as cp
from lib.spark import *
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



spark_connect = SparkConnect(spark_cores=12,spark_memory="14g")
sqlContext = spark_connect.get_sql_context()


#Read the Prometheus JSON BZip data
# jsonFile = sqlContext.read.option("multiline", True).option("mode", "PERMISSIVE").json("s3a://DH-DEV-PROMETHEUS-BACKUP/prometheus-openshift-devops-monitor.1b7d.free-stg.openshiftapps.com/"+metric_name+"/")
prom_host = "prometheus-openshift-devops-monitor.1b7d.free-stg.openshiftapps.com"
jsonUrl = "s3a://" + os.getenv('DH_CEPH_BUCKET')+ "/" + prom_host + "/" + metric_name
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






if label != "":
    select_labels = ['metric.' + label]
else:
    select_labels = []

# get data and format
data = extract_from_json(jsonFile, metric_name, select_labels, where_labels)

data.count()
data.show()





data = get_data_in_timeframe(data, start_time, end_time)
data.show()



calculate_sample_rate(data)



calculate_vals_per_label(data, label)

from pyspark.sql.window import Window



max_delta, min_delta, mean_delta, data = get_deltas(data)
data.show()
print("Max delta:", max_delta)
print("Min delta:", min_delta)
print("Mean delta:", mean_delta)

# rhmax = current value / max(values)
# if rhmax (of new point) >> 1, we can assume that the point is an anomaly


result = get_rhmax(data)
result.show()



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




mean, var, stddev, median = get_stats(data)

print("\tMean(values): ", mean)
print("\tVariance(values): ", var)
print("\tStddev(values): ", stddev)
print("\tMedian(values): ", median)


OP_TYPE = 'list_images'
data = data.filter(data.operation_type == OP_TYPE)

data_pd = data.toPandas()
del data
# be sure to stop the Spark Session to conserve resources
spark_connect.stop_spark()

# Delete Spark Cluster
# import subprocess
# subprocess.call("$HOME/delete_spark.sh")
os.system("source $HOME/delete_spark.sh")

from fbprophet import Prophet

#temp_frame = get_filtered_op_frame(OP_TYPE)
data_pd = data_pd.set_index(data_pd.timestamp)
data_pd = data_pd[['timestamp','values']]


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
forecast['timestamp'] = forecast['ds']
forecast = forecast[['timestamp','yhat','yhat_lower','yhat_upper']]

# Store Forecast to CEPH
session = cp()
object_path = "Predictions" + "/" + prom_host + "/" + metric_name + "_" + (start_time.strftime("%Y%m%d%H%M")) + "_" + (end_time.strftime("%Y%m%d%H%M")) + ".json"
print(session.store_data(name = metric_name, object_path = object_path, values = forecast.to_json()))


import pandas as pd

forecast = forecast.set_index(forecast.timestamp)
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

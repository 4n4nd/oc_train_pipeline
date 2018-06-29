import pandas as pd
import json
import numpy as np
import sys
import pyspark
import os
from fbprophet import Prophet
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

OP_TYPE = 'list_images'

SPARK_MASTER="spark://" + os.getenv('OSHINKO_CLUSTER_NAME') + ":7077"
#Set the configuration
conf = pyspark.SparkConf().setAppName('Ceph S3 Prometheus JSON Reader').setMaster(SPARK_MASTER)

#Set the Spark cluster connection
sc = pyspark.SparkContext.getOrCreate(conf)

#Set the Hadoop configurations to access Ceph S3
sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", os.getenv('DH_CEPH_KEY'))
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", os.getenv('DH_CEPH_SECRET'))
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", os.getenv('DH_CEPH_HOST'))

#Get the SQL context
sqlContext = pyspark.SQLContext(sc)

metric_name = os.getenv('PROM_METRIC_NAME')
#Read the Prometheus JSON BZip data
jsonFile = sqlContext.read.option("multiline", True).option("mode", "PERMISSIVE").json("s3a://DH-DEV-PROMETHEUS-BACKUP/prometheus-openshift-devops-monitor.1b7d.free-stg.openshiftapps.com/"+metric_name+"/")

#Display the schema of the file
print('Display schema:')
jsonFile.printSchema()

#Register the created SchemaRDD as a temporary table.
jsonFile.registerTempTable(metric_name)

#select time, value, operation_type from kubelet_docker_operations_latency_microseconds where quantile='0.9' and hostname='free-stg-master-03fb6'

#Filter the results into a data frame
data = sqlContext.sql("SELECT values, metric.operation_type FROM " + metric_name+" WHERE metric.quantile='0.9' AND metric.hostname='free-stg-master-03fb6'")

data.show()

data.select('operation_type').distinct().collect()
sliced_frame = data.filter(data.operation_type == OP_TYPE)
del data

reduced_sliced_frame = sqlContext.createDataFrame(sliced_frame.head(int(sliced_frame.count()/3)), sliced_frame.schema)
del sliced_frame
data_pd = reduced_sliced_frame.toPandas()
sc.stop()

df2 = pd.DataFrame(columns = ['utc_timestamp','value', 'operation_type'])
#df2 ='
for op in set(data_pd['operation_type']):
    dict_raw = data_pd[data_pd['operation_type'] == op]['values']
    list_raw = []
    for key in dict_raw.keys():
        list_raw.extend(dict_raw[key])
    temp_frame = pd.DataFrame(list_raw, columns = ['utc_timestamp','value'])
    temp_frame['operation_type'] = op

    df2 = df2.append(temp_frame)

df2 = df2[df2['value'] != 'NaN']
df2['value'] = df2['value'].apply(lambda a: int(a))
df2['timestamp'] = df2['utc_timestamp'].apply(lambda a : datetime.fromtimestamp(int(a)))
df2.head()

df2.reset_index(inplace =True)
del df2['index']
df2['operation_type'].unique()

def get_filtered_op_frame(op_type):
    temp = df2[df2.operation_type == op_type]
    temp = temp.sort_values(by='timestamp')
    return temp

operation_type_value = {}
for temp in list(df2.operation_type.unique()):
    operation_type_value[temp] = get_filtered_op_frame(temp)['value']

for temp in operation_type_value.keys():
    print("Mean of: ",temp, " - ", np.mean(operation_type_value[temp]))

for temp in operation_type_value.keys():
    print("Variance of: ",temp, " - ", np.var(operation_type_value[temp]))

for temp in operation_type_value.keys():
    print("Standard Deviation of: ",temp, " - ", np.std(operation_type_value[temp]))

for temp in operation_type_value.keys():
    print("Median of: ",temp, " - ", np.median(operation_type_value[temp]))

df_whisker =  df2
df_whisker['log_transformed_value'] = np.log(df2['value'])

df_whisker.head()

for temp in operation_type_value.keys():
    #fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,12))
    temp_frame = get_filtered_op_frame(temp)
    temp_frame = temp_frame.set_index(temp_frame.timestamp)
    temp_frame = temp_frame[['log_transformed_value']]
    temp_frame.plot(figsize=(15,12),title=temp)

temp_frame = get_filtered_op_frame(OP_TYPE)
temp_frame = temp_frame.set_index(temp_frame.timestamp)
temp_frame = temp_frame[['timestamp','value']]

train_frame['y'] = train_frame['value']
train_frame['ds'] = train_frame['timestamp']

m = Prophet()
m.fit(train_frame)

future = m.make_future_dataframe(periods= int(len(test_frame) * 1.1),freq= '1MIN')
forecast = m.predict(future)
forecast.head()

forecasted_features = ['ds','yhat','yhat_lower','yhat_upper']

forecast = forecast[forecasted_features]
forecast.head()

forecast['timestamp'] = forecast['ds']
joined_frame = pd.merge(test_frame, forecast, how='inner', on=['timestamp'],left_index=True)
joined_frame.head()

joined_frame.count()
joined_frame['residuals'] = joined_frame['value'] - joined_frame['yhat']

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


joined_frame.set_index(joined_frame['timestamp'],inplace= True)
joined_frame2 = joined_frame[:int(0.3 * len(joined_frame))]
test_frame['timestamp'].max()

def trace_insight(op_type, return_response = True):
    df_temp = df2[df2.operation_type == op_type]
    max_value = df_temp['value'].max()
    min_value = df_temp['value'].min()
    median_value = np.median(df_temp['value'])
    mean = np.mean(df_temp['value'])
    std = np.std(df_temp['value'])
    alert_level_upper = mean + 2 * std
    if return_response:
        return json.dumps({'min':str(min_value), 'max':str(max_value), 'median':str(median_value), 'mean': str(mean), 'std' :str(std), 'alert_level': str(alert_level_upper)})
    print ('min',min_value, 'max',max_value,'median',median_value)


trace_insight('create_container')
trace_insight('list_containers')

df_list_containers = get_filtered_op_frame('list_containers')
df_create_container = get_filtered_op_frame('create_container')

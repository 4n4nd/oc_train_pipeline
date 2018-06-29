import pandas as pd
import json
import numpy as np
import sys
import pyspark
import os
from fbprophet import Prophet

import warnings
warnings.filterwarnings('ignore')

OP_TYPE = 'list_images'

#Set the configuration
conf = pyspark.SparkConf().setAppName('Ceph S3 Prometheus JSON Reader').setMaster('local[2]')

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



def get_filtered_op_frame(op_type):
    temp = df2[df2.operation_type == op_type]
    temp = temp.sort_values(by='timestamp')
    return temp

temp_frame = get_filtered_op_frame(OP_TYPE)
temp_frame = temp_frame.set_index(temp_frame.timestamp)
temp_frame = temp_frame[['timestamp','value']]

train_frame = temp_frame[0 : int(0.7*len(temp_frame))]
test_frame = temp_frame[int(0.7*len(temp_frame)) : ]

print(len(train_frame), len(test_frame), len(temp_frame))

train_frame['y'] = train_frame['value']
train_frame['ds'] = train_frame['timestamp']

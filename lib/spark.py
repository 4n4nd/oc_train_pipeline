import pyspark
import os
import random
import string
import datetime
import json

import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import TimestampType
from pyspark.sql.window import Window

class SparkConnect:
    def __init__(self, spark_master=None, app_name=None, spark_cores=2, spark_memory="1g", ceph_access_key=None, ceph_secret_key=None, ceph_host_url=None):

        if not spark_master:
            if os.getenv('SPARK_LOCAL')=="True":
                spark_master='local[2]'
                spark_cores = 2
                spark_memory = "1g"
                print("Using local spark")
                pass
            else:
                spark_master="spark://" + os.getenv('OSHINKO_CLUSTER_NAME') + ":7077"
        pass
        if not app_name:
            inst = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            app_name = inst + ' - Ephemeral Spark Application'
        #Set the configuration
        print("Application Name: ", app_name)


        self.spark_settings = {
            'spark_master': spark_master,
            'app_name': app_name,
            'spark_cores': spark_cores,
            'spark_memory': spark_memory
        }

        conf = pyspark.SparkConf().setAppName(self.spark_settings['app_name']).setMaster(spark_master)

        conf.set("spark.cores.max", str(self.spark_settings['spark_cores']))
        conf.set("spark.executor.memory", self.spark_settings['spark_memory'])

        #Set the Spark cluster connection
        self.sc = pyspark.SparkContext.getOrCreate(conf)

        #Set the Hadoop configurations to access Ceph S3
        self.sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", os.getenv('DH_CEPH_KEY',ceph_access_key))
        self.sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", os.getenv('DH_CEPH_SECRET',ceph_secret_key))
        self.sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", os.getenv('DH_CEPH_HOST',ceph_host_url))

        #Get the SQL context
        self.sqlContext = pyspark.SQLContext(self.sc)

    def get_spark_master(self):
        return self.spark_settings['spark_master']

    def get_spark_context(self):
        return self.sc
        pass

    def get_sql_context(self):
        return self.sqlContext
        pass

    def stop_spark(self):
        self.sc.stop()
        pass

def convert_timestamp(t):
    # create function to convert POSIX timestamp to local date
    return datetime.fromtimestamp(float(t))

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
    # data = sqlContext.createDataFrame(data.head(1000), data.schema)

    return format_df(data)

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

def calculate_vals_per_label(df, df_label):
    # new df with vals per label
    df.groupBy(df_label).count().show()

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

def get_rhmax(df):
    real_max = df.agg(F.max(F.col("values"))).head()[0]
    result = df.withColumn("rhmax", df["values"]/real_max)
    return result

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

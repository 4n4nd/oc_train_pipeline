import pyspark
import os
import random
import string

class SparkConnect:
    def __init__(self, spark_master=None, app_name=None, spark_cores=2, spark_memory="1g", ceph_access_key=None, ceph_secret_key=None, ceph_host_url=None):

        if not spark_master:
            if os.getenv('SPARK_LOCAL')=="True":
                spark_master='local[2]'
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

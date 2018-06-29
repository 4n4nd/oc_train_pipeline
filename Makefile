APP_NAME='pyspark-prophet-test' # This is also the gitlab webhook secret as in oshinko image

REPO_URI='https://gitlab.cee.redhat.com/asanmukh/oc_train_pipeline.git'
CEPH_ACCESS_KEY='STORAGE_ACCESS_KEY'
CEPH_SECRET_KEY='STORAGE_SECRET_KEY'
CEPH_HOST_URL='STORAGE_ENDPOINT_URL'

METRIC_NAME='kubelet_docker_operations_latency_microseconds'
METRIC_TIME_FRAME=1234

oc_add_template:
	oc create -f ./resources.yaml

oc_replace_template:
	oc replace -f ./resources.yaml

oc_deploy:
	oc new-app --template oshinko-python-spark-build-dc  \
    -p APPLICATION_NAME=${APP_NAME} \
    -p GIT_URI=${REPO_URI} \
		-p DH_CEPH_KEY=${CEPH_ACCESS_KEY} \
		-p DH_CEPH_SECRET=${CEPH_SECRET_KEY} \
		-p DH_CEPH_HOST=${CEPH_HOST_URL} \
		-p PROM_METRIC_NAME=${METRIC_NAME} \
		-p METRIC_TIME_FRAME=${METRIC_TIME_FRAME} \

oc_delete:
	oc delete all -l app=${APP_NAME}

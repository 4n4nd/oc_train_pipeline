source $APP_ROOT/etc/generate_container_user
CA="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

if [ "$KUBERNETES_SERVICE_PORT" -eq 443 ]; then
    KUBE_SCHEME="https"
else
    KUBE_SCHEME="http"
fi
KUBE="$KUBE_SCHEME://$KUBERNETES_SERVICE_HOST:$KUBERNETES_SERVICE_PORT"

SA=`cat /var/run/secrets/kubernetes.io/serviceaccount/token`
NS=`cat /var/run/secrets/kubernetes.io/serviceaccount/namespace`
CLI_ARGS="--certificate-authority=$CA --server=$KUBE --token=$SA --namespace=$NS"

$CLI delete $OSHINKO_CLUSTER_NAME $CLI_ARGS

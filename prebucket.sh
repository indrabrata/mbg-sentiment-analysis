#!/bin/sh

MC_ALIAS="localminio"
BUCKET="mlflow"

mc alias set $MC_ALIAS http://$MINIO_HOST:$MINIO_PORT $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD

if mc ls $MC_ALIAS/$BUCKET >/dev/null 2>&1; then
    echo "Bucket '$BUCKET' already exists"
else
    echo "Creating bucket '$BUCKET'..."
    mc mb $MC_ALIAS/$BUCKET
fi

echo "Bucket '$BUCKET' is ready."
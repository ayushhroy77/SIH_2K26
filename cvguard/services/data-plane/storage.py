"""MinIO (S3-compatible) blob store abstraction for air-gapped image retention."""

from __future__ import annotations

import io
import logging
import os
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger("cvguard.dataplane.storage")

DEFAULT_IMAGE_BUCKET = "images"


class MinIOStorage:
    """MinIO storage client manager with automated bucket provisioning."""

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        secure: bool = False,
    ) -> None:
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = access_key or os.getenv("MINIO_ROOT_USER", "minio_admin")
        self.secret_key = secret_key or os.getenv(
            "MINIO_ROOT_PASSWORD", "minio_dev_secret_change_me_in_prod"
        )
        self.secure = secure
        self._client: Minio | None = None

    @property
    def client(self) -> Minio:
        """Lazy initialization of the Minio client."""
        if self._client is None:
            self._client = Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
        return self._client

    def ensure_bucket(self, bucket_name: str = DEFAULT_IMAGE_BUCKET) -> None:
        """Ensure that the target object storage bucket exists, creating it if missing."""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info("Created MinIO bucket '%s'", bucket_name)
        except S3Error as exc:
            logger.error("Failed to ensure MinIO bucket '%s': %s", bucket_name, exc)
            raise

    def put_image(
        self,
        object_name: str,
        data: bytes,
        content_type: str = "image/jpeg",
        bucket_name: str = DEFAULT_IMAGE_BUCKET,
    ) -> str:
        """Store image bytes in MinIO bucket under object_name."""
        self.ensure_bucket(bucket_name)
        stream = io.BytesIO(data)
        self.client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=stream,
            length=len(data),
            content_type=content_type,
        )
        logger.debug("Persisted object '%s' in bucket '%s' (%d bytes)", object_name, bucket_name, len(data))
        return object_name

    def get_image(
        self,
        object_name: str,
        bucket_name: str = DEFAULT_IMAGE_BUCKET,
    ) -> tuple[bytes, str]:
        """Retrieve raw object bytes and content_type from MinIO."""
        response = None
        try:
            response = self.client.get_object(bucket_name=bucket_name, object_name=object_name)
            content_type = response.headers.get("content-type", "image/jpeg")
            data = response.read()
            return data, content_type
        finally:
            if response is not None:
                response.close()
                response.release_conn()

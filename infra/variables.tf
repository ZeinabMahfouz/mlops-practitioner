variable "postgres_password" {
  type        = string
  description = "Password for PostgreSQL backend"
  default     = "postgres"
}

variable "minio_access_key" {
  type        = string
  description = "MinIO access key"
  default     = "admin"
}

variable "minio_secret_key" {
  type        = string
  description = "MinIO secret key"
  default     = "password"
}

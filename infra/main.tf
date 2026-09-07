resource "docker_network" "mlops_net" {
  name = "mlops-net"
}

resource "docker_image" "postgres" {
  name = "postgres:14"
}

resource "docker_image" "minio" {
  name = "minio/minio:latest"
}

resource "docker_container" "postgres" {
  name    = "mlflow_db"
  image   = docker_image.postgres.image_id
  restart = "always"
  env = [
    "POSTGRES_USER=postgres",
    "POSTGRES_PASSWORD=${var.postgres_password}",
    "POSTGRES_DB=mlflow"
  ]
  networks_advanced {
    name = docker_network.mlops_net.name
  }
}

resource "docker_container" "minio" {
  name    = "minio"
  image   = docker_image.minio.image_id
  restart = "always"
  command = ["server", "/data", "--console-address", ":9001"]
  env = [
    "MINIO_ROOT_USER=${var.minio_access_key}",
    "MINIO_ROOT_PASSWORD=${var.minio_secret_key}"
  ]
  ports {
    internal = 9000
    external = 9000
  }
  ports {
    internal = 9001
    external = 9001
  }
  networks_advanced {
    name = docker_network.mlops_net.name
  }
}

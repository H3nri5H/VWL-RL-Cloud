# Cloud Storage Buckets for ML Models

resource "google_storage_bucket" "ml_models" {
  name     = "${var.project_id}-ml-models"
  location = "EU"

  uniform_bucket_level_access = true

  # Versioning for model history
  versioning {
    enabled = true
  }

  # Lifecycle: Delete old models after 90 days
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    purpose     = "ml-models"
  }
}

# Bucket for training logs
resource "google_storage_bucket" "training_logs" {
  name     = "${var.project_id}-training-logs"
  location = "EU"

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 30 # Keep logs for 30 days
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    purpose     = "logs"
  }
}

output "models_bucket" {
  value = google_storage_bucket.ml_models.name
}

output "logs_bucket" {
  value = google_storage_bucket.training_logs.name
}

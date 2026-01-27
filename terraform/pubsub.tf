# Pub/Sub for Event-Driven Architecture

# Topic: Training Events
resource "google_pubsub_topic" "training_events" {
  name = "training-events-${var.environment}"

  labels = {
    environment = var.environment
  }
}

# Subscription: Backend receives training completion events
resource "google_pubsub_subscription" "backend_training_sub" {
  name  = "backend-training-sub-${var.environment}"
  topic = google_pubsub_topic.training_events.id

  # Keep messages for 7 days
  message_retention_duration = "604800s"

  # Acknowledge deadline
  ack_deadline_seconds = 60

  # Retry policy
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
}

# Topic: Model Updates
resource "google_pubsub_topic" "model_updates" {
  name = "model-updates-${var.environment}"

  labels = {
    environment = var.environment
  }
}

output "training_topic" {
  value = google_pubsub_topic.training_events.id
}

output "backend_subscription" {
  value = google_pubsub_subscription.backend_training_sub.id
}

# Terraform Variables

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "europe-west1"
}

variable "environment" {
  description = "Environment (dev/prod)"
  type        = string
  default     = "dev"
}

variable "gke_machine_type" {
  description = "GKE Node machine type"
  type        = string
  default     = "e2-standard-2"
}

variable "gke_min_nodes" {
  description = "Minimum GKE nodes"
  type        = number
  default     = 1
}

variable "gke_max_nodes" {
  description = "Maximum GKE nodes"
  type        = number
  default     = 5
}

# Google Kubernetes Engine (GKE) Cluster

resource "google_container_cluster" "vwl_rl_cluster" {
  name     = "vwl-rl-cluster-${var.environment}"
  location = var.region

  # Start with 1 node, scale up/down automatically
  initial_node_count       = 1
  remove_default_node_pool = true

  # Network config
  network    = "default"
  subnetwork = "default"

  # Release channel for auto-updates
  release_channel {
    channel = "REGULAR"
  }
}

# Node Pool for workloads
resource "google_container_node_pool" "primary_nodes" {
  name       = "primary-pool"
  cluster    = google_container_cluster.vwl_rl_cluster.id
  node_count = 2

  autoscaling {
    min_node_count = 1
    max_node_count = 5
  }

  node_config {
    machine_type = "e2-standard-2" # 2 vCPU, 8GB RAM
    disk_size_gb = 50

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      app         = "vwl-rl"
    }
  }
}

output "gke_cluster_name" {
  value = google_container_cluster.vwl_rl_cluster.name
}

output "gke_cluster_endpoint" {
  value = google_container_cluster.vwl_rl_cluster.endpoint
}

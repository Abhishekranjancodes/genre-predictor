# Ansible Configuration Management

This directory contains Ansible playbooks and roles for automating the deployment of the **RoBERTa Genre Classifier** MLOps project.

## Directory Structure

```
ansible/
├── ansible.cfg                  # Ansible configuration
├── inventory/
│   └── hosts.ini                # Inventory file (localhost)
├── site.yml                     # Full pipeline: build + deploy + ELK + verify
├── deploy.yml                   # K8s deploy + ELK + verify (skip build)
└── roles/
    ├── docker/                  # Role: Docker install, build & push images
    │   ├── tasks/main.yml
    │   ├── defaults/main.yml
    │   └── meta/main.yml
    ├── k8s/                     # Role: Kubernetes deployment with HPA
    │   ├── tasks/main.yml
    │   ├── defaults/main.yml
    │   └── meta/main.yml
    ├── elk/                     # Role: ELK Stack deployment (logging)
    │   ├── tasks/main.yml
    │   ├── defaults/main.yml
    │   └── meta/main.yml
    └── monitoring/              # Role: Health checks & deployment verification
        ├── tasks/main.yml
        ├── defaults/main.yml
        └── meta/main.yml
```

## Playbooks

| Playbook     | Purpose                                                              |
|------------- |----------------------------------------------------------------------|
| `site.yml`   | Full pipeline — build Docker images, deploy to K8s, set up ELK, verify |
| `deploy.yml` | K8s only — deploy manifests, set up ELK, verify (images already built) |

## Usage

### Full Pipeline (build + deploy + ELK + verify)

```bash
cd ansible
DOCKER_HUB_PASSWORD=<your-password> ansible-playbook site.yml
```

### Deploy Only (skip Docker build)

```bash
cd ansible
ansible-playbook deploy.yml
```

### Run Specific Roles via Tags

```bash
# Only build and push Docker images
ansible-playbook site.yml --tags docker

# Only deploy to Kubernetes
ansible-playbook site.yml --tags k8s

# Only deploy ELK Stack
ansible-playbook site.yml --tags elk

# Only run health checks
ansible-playbook site.yml --tags monitoring
```

## Roles

### `docker`
- Installs Docker Engine if not present
- Builds `roberta-backend` and `roberta-frontend` images
- Pushes images to Docker Hub

### `k8s`
- Applies Kubernetes deployment manifests (`k8s/deployment.yaml`)
- Performs rolling restart of deployments
- Applies Horizontal Pod Autoscaler (`k8s/hpa.yaml`)
- Waits for rollouts to complete

### `elk`
- Deploys Logstash pipeline and settings ConfigMaps
- Deploys Elasticsearch (single-node), Logstash, and Kibana
- Deploys Filebeat DaemonSet with RBAC for log collection
- Waits for all ELK components to be ready
- Verifies Elasticsearch cluster health
- Log index pattern: `roberta-logs-*`

### `monitoring`
- Verifies pod and service status
- Runs health check against the backend `/health` endpoint
- Displays resource usage and deployment summary

## ELK Stack Architecture

```
┌──────────────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────────┐
│  Backend/Frontend│────▶│   Filebeat   │────▶│    Logstash     │────▶│  Elastic │
│  (JSON logs)     │     │  (DaemonSet) │     │  (Processing)  │     │  search  │
└──────────────────┘     └──────────────┘     └────────────────┘     └────┬─────┘
                                                                         │
                                                                    ┌────▼─────┐
                                                                    │  Kibana  │
                                                                    │ (UI:5601)│
                                                                    └──────────┘
```

### Accessing Kibana

After deployment, access Kibana at: `http://<node-ip>:5601`

1. Navigate to **Stack Management → Data Views**
2. Create a data view with pattern `roberta-logs-*`
3. Go to **Discover** to explore application logs
4. Build dashboards to visualize predictions, response times, and error rates

## Jenkins Integration

The `Jenkinsfile` calls `ansible-playbook deploy.yml` in the deployment stage, which automatically deploys both the application and the ELK logging stack.

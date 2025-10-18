#!/usr/bin/env python3
"""
Generate deployment configuration files for different container orchestration platforms.

This script creates optimized deployment configurations with proper health checks
for Kubernetes, Docker Swarm, and Docker Compose.
"""

import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

from health_check_config import HealthCheckManager, HealthCheckType


def generate_kubernetes_deployment(output_dir: Path, service_name: str = "hmm-service") -> None:
    """Generate Kubernetes deployment configuration."""
    
    manager = HealthCheckManager()
    
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": service_name,
            "labels": {
                "app": service_name,
                "version": "v1.0.0",
                "component": "inference"
            }
        },
        "spec": {
            "replicas": 3,
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxSurge": 1,
                    "maxUnavailable": 0
                }
            },
            "selector": {
                "matchLabels": {
                    "app": service_name
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": service_name,
                        "version": "v1.0.0",
                        "component": "inference"
                    }
                },
                "spec": {
                    "containers": [{
                        "name": service_name,
                        "image": f"imp/{service_name}:v1.0.0",
                        "ports": [
                            {"containerPort": 8000, "name": "http", "protocol": "TCP"}
                        ],
                        "env": [
                            {"name": "HMM_SERVICE_HOST", "value": "0.0.0.0"},
                            {"name": "HMM_SERVICE_PORT", "value": "8000"},
                            {"name": "HMM_LOG_LEVEL", "value": "INFO"},
                            {"name": "HMM_LOG_FORMAT", "value": "json"},
                            {"name": "ORCHESTRATION_TYPE", "value": "kubernetes"},
                            {"name": "MINIO_ENDPOINT", "value": "minio-service:9000"},
                            {"name": "HMM_CACHE_SIZE", "value": "1000"},
                            {"name": "HMM_CACHE_TTL", "value": "300"},
                            {"name": "HMM_MAX_CONCURRENT_REQUESTS", "value": "100"}
                        ],
                        "resources": {
                            "requests": {
                                "memory": "256Mi",
                                "cpu": "250m"
                            },
                            "limits": {
                                "memory": "512Mi",
                                "cpu": "500m"
                            }
                        },
                        "livenessProbe": manager.get_kubernetes_probe_config(HealthCheckType.LIVENESS),
                        "readinessProbe": manager.get_kubernetes_probe_config(HealthCheckType.READINESS),
                        "startupProbe": manager.get_kubernetes_probe_config(HealthCheckType.STARTUP),
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "readOnlyRootFilesystem": True,
                            "allowPrivilegeEscalation": False,
                            "capabilities": {
                                "drop": ["ALL"]
                            }
                        },
                        "volumeMounts": [
                            {"name": "tmp", "mountPath": "/tmp"},
                            {"name": "cache", "mountPath": "/app/cache"}
                        ]
                    }],
                    "securityContext": {
                        "fsGroup": 1000
                    },
                    "volumes": [
                        {"name": "tmp", "emptyDir": {}},
                        {"name": "cache", "emptyDir": {"sizeLimit": "100Mi"}}
                    ],
                    "terminationGracePeriodSeconds": 30
                }
            }
        }
    }
    
    # Service configuration
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": service_name,
            "labels": {
                "app": service_name
            }
        },
        "spec": {
            "type": "ClusterIP",
            "ports": [{
                "port": 8000,
                "targetPort": 8000,
                "protocol": "TCP",
                "name": "http"
            }],
            "selector": {
                "app": service_name
            }
        }
    }
    
    # HPA configuration
    hpa = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {
            "name": f"{service_name}-hpa"
        },
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": service_name
            },
            "minReplicas": 2,
            "maxReplicas": 10,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": 70
                        }
                    }
                },
                {
                    "type": "Resource",
                    "resource": {
                        "name": "memory",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": 80
                        }
                    }
                }
            ]
        }
    }
    
    # Write files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "kubernetes-deployment.yaml", "w") as f:
        yaml.dump_all([deployment, service, hpa], f, default_flow_style=False)
    
    print(f"Generated Kubernetes deployment configuration in {output_dir}/kubernetes-deployment.yaml")


def generate_docker_swarm_config(output_dir: Path, service_name: str = "hmm-service") -> None:
    """Generate Docker Swarm configuration."""
    
    manager = HealthCheckManager()
    healthcheck_config = manager.get_docker_healthcheck_config()
    
    swarm_config = {
        "version": "3.8",
        "services": {
            service_name: {
                "image": f"imp/{service_name}:v1.0.0",
                "healthcheck": healthcheck_config,
                "deploy": {
                    "replicas": 3,
                    "resources": {
                        "limits": {
                            "memory": "512M",
                            "cpus": "0.5"
                        },
                        "reservations": {
                            "memory": "256M",
                            "cpus": "0.25"
                        }
                    },
                    "update_config": {
                        "parallelism": 1,
                        "delay": "10s",
                        "failure_action": "rollback",
                        "monitor": "60s",
                        "max_failure_ratio": 0.1,
                        "order": "start-first"
                    },
                    "rollback_config": {
                        "parallelism": 1,
                        "delay": "10s",
                        "failure_action": "pause",
                        "monitor": "60s",
                        "max_failure_ratio": 0.1,
                        "order": "stop-first"
                    },
                    "restart_policy": {
                        "condition": "on-failure",
                        "delay": "5s",
                        "max_attempts": 3,
                        "window": "120s"
                    },
                    "placement": {
                        "constraints": ["node.role == worker"],
                        "preferences": [{"spread": "node.id"}]
                    }
                },
                "ports": [
                    {"target": 8000, "published": 8000, "protocol": "tcp", "mode": "ingress"}
                ],
                "environment": [
                    "HMM_SERVICE_HOST=0.0.0.0",
                    "HMM_SERVICE_PORT=8000",
                    "HMM_LOG_LEVEL=INFO",
                    "HMM_LOG_FORMAT=json",
                    "ORCHESTRATION_TYPE=docker-swarm",
                    "MINIO_ENDPOINT=minio:9000",
                    "HMM_CACHE_SIZE=1000",
                    "HMM_CACHE_TTL=300",
                    "HMM_MAX_CONCURRENT_REQUESTS=100"
                ],
                "networks": ["hmm-network"],
                "depends_on": ["minio"]
            },
            "minio": {
                "image": "minio/minio:latest",
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"],
                    "interval": "30s",
                    "timeout": "20s",
                    "retries": 3,
                    "start_period": "10s"
                },
                "deploy": {
                    "replicas": 1,
                    "resources": {
                        "limits": {"memory": "256M", "cpus": "0.25"},
                        "reservations": {"memory": "128M", "cpus": "0.1"}
                    },
                    "restart_policy": {
                        "condition": "on-failure",
                        "delay": "5s",
                        "max_attempts": 3
                    },
                    "placement": {
                        "constraints": ["node.role == manager"]
                    }
                },
                "command": "server /data --console-address \":9001\"",
                "environment": [
                    "MINIO_ROOT_USER=minioadmin",
                    "MINIO_ROOT_PASSWORD=minioadmin123"
                ],
                "volumes": ["minio_data:/data"],
                "networks": ["hmm-network"],
                "ports": [
                    {"target": 9000, "published": 9000, "protocol": "tcp"},
                    {"target": 9001, "published": 9001, "protocol": "tcp"}
                ]
            }
        },
        "networks": {
            "hmm-network": {
                "driver": "overlay",
                "attachable": True
            }
        },
        "volumes": {
            "minio_data": {
                "driver": "local"
            }
        }
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "docker-swarm-stack.yml", "w") as f:
        yaml.dump(swarm_config, f, default_flow_style=False)
    
    print(f"Generated Docker Swarm configuration in {output_dir}/docker-swarm-stack.yml")


def generate_docker_compose_config(output_dir: Path, service_name: str = "hmm-service") -> None:
    """Generate Docker Compose configuration."""
    
    manager = HealthCheckManager()
    healthcheck_config = manager.get_docker_healthcheck_config()
    
    compose_config = {
        "version": "3.8",
        "services": {
            service_name: {
                "build": {
                    "context": ".",
                    "target": "production"
                },
                "ports": ["8000:8000"],
                "environment": [
                    "HMM_SERVICE_HOST=0.0.0.0",
                    "HMM_SERVICE_PORT=8000",
                    "HMM_SERVICE_DEBUG=false",
                    "HMM_LOG_LEVEL=INFO",
                    "HMM_LOG_FORMAT=json",
                    "ORCHESTRATION_TYPE=docker-compose",
                    "MINIO_ENDPOINT=minio:9000",
                    "MINIO_ACCESS_KEY=minioadmin",
                    "MINIO_SECRET_KEY=minioadmin123",
                    "MINIO_BUCKET=hmm-artifacts",
                    "MINIO_SECURE=false",
                    "HMM_DEFAULT_EXPERIMENT_ID=development_hmm",
                    "HMM_CACHE_SIZE=1000",
                    "HMM_CACHE_TTL=300",
                    "HMM_MAX_CONCURRENT_REQUESTS=100"
                ],
                "depends_on": {
                    "minio": {"condition": "service_healthy"}
                },
                "restart": "unless-stopped",
                "deploy": {
                    "resources": {
                        "limits": {"memory": "512M", "cpus": "0.5"},
                        "reservations": {"memory": "256M", "cpus": "0.25"}
                    }
                },
                "healthcheck": healthcheck_config,
                "networks": ["hmm-network"]
            },
            "minio": {
                "image": "minio/minio:latest",
                "ports": ["9000:9000", "9001:9001"],
                "environment": [
                    "MINIO_ROOT_USER=minioadmin",
                    "MINIO_ROOT_PASSWORD=minioadmin123"
                ],
                "command": "server /data --console-address \":9001\"",
                "volumes": ["minio_data:/data"],
                "restart": "unless-stopped",
                "deploy": {
                    "resources": {
                        "limits": {"memory": "256M", "cpus": "0.25"}
                    }
                },
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"],
                    "interval": "30s",
                    "timeout": "20s",
                    "retries": 3,
                    "start_period": "10s"
                },
                "networks": ["hmm-network"]
            }
        },
        "volumes": {
            "minio_data": {"driver": "local"}
        },
        "networks": {
            "hmm-network": {"driver": "bridge"}
        }
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "docker-compose.production.yml", "w") as f:
        yaml.dump(compose_config, f, default_flow_style=False)
    
    print(f"Generated Docker Compose configuration in {output_dir}/docker-compose.production.yml")


def generate_health_check_examples(output_dir: Path) -> None:
    """Generate health check configuration examples."""
    
    manager = HealthCheckManager()
    
    examples = {
        "kubernetes_probes": {
            "liveness": manager.get_kubernetes_probe_config(HealthCheckType.LIVENESS),
            "readiness": manager.get_kubernetes_probe_config(HealthCheckType.READINESS),
            "startup": manager.get_kubernetes_probe_config(HealthCheckType.STARTUP)
        },
        "docker_healthcheck": manager.get_docker_healthcheck_config(),
        "environment_variables": {
            "kubernetes": {
                "ORCHESTRATION_TYPE": "kubernetes",
                "HEALTH_CHECK_MODE": "kubernetes"
            },
            "docker_swarm": {
                "ORCHESTRATION_TYPE": "docker-swarm",
                "HEALTH_CHECK_MODE": "docker"
            },
            "docker_compose": {
                "ORCHESTRATION_TYPE": "docker-compose",
                "HEALTH_CHECK_MODE": "docker"
            }
        }
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "health-check-examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"Generated health check examples in {output_dir}/health-check-examples.json")


def main():
    """Main function to generate deployment configurations."""
    
    parser = argparse.ArgumentParser(description="Generate deployment configurations for HMM Microservice")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("./deployment-configs"),
                       help="Output directory for generated configurations")
    parser.add_argument("--service-name", "-n", default="hmm-service",
                       help="Service name to use in configurations")
    parser.add_argument("--platform", "-p", choices=["kubernetes", "docker-swarm", "docker-compose", "all"],
                       default="all", help="Platform to generate configurations for")
    
    args = parser.parse_args()
    
    print(f"Generating deployment configurations for {args.platform}...")
    
    if args.platform in ["kubernetes", "all"]:
        generate_kubernetes_deployment(args.output_dir / "kubernetes", args.service_name)
    
    if args.platform in ["docker-swarm", "all"]:
        generate_docker_swarm_config(args.output_dir / "docker-swarm", args.service_name)
    
    if args.platform in ["docker-compose", "all"]:
        generate_docker_compose_config(args.output_dir / "docker-compose", args.service_name)
    
    # Always generate health check examples
    generate_health_check_examples(args.output_dir)
    
    print("\nDeployment configuration generation complete!")
    print(f"Files generated in: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review and customize the generated configurations")
    print("2. Update environment variables for your specific setup")
    print("3. Deploy using the appropriate orchestration platform")
    print("4. Monitor health checks and adjust thresholds as needed")


if __name__ == "__main__":
    main()
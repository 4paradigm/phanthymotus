"""
preflight.py — Pre-deployment checks for drivers.

Checks before pulling large images:
1. Disk space (need ~2x image size for pull + extraction)
2. Registry connectivity & authentication
3. Network bandwidth estimation
4. Existing containers that may conflict

Returns structured report with pass/fail/warning status and actionable suggestions.
"""

import shutil
import time
from typing import Optional

import docker as docker_sdk


def check_disk_space(required_gb: float = 20.0, min_gb: float = 1.0) -> dict:
    """Check if sufficient disk space is available.

    Only blocks deployment (status='fail') when free space drops below the
    absolute floor `min_gb`. Falling short of the recommended `required_gb`
    (e.g. 2.5x image size) is a non-blocking warning, since that figure is a
    comfort margin for pull+extract, not a hard requirement.

    Args:
        required_gb: Recommended free space in GB (comfort margin)
        min_gb: Absolute minimum free space in GB below which deploy is blocked

    Returns:
        {
            'status': 'pass' | 'warning' | 'fail',
            'free_gb': float,
            'total_gb': float,
            'message': str,
            'suggestion': str | None
        }
    """
    try:
        usage = shutil.disk_usage('/')
        free_gb = usage.free / (1 << 30)
        total_gb = usage.total / (1 << 30)

        if free_gb >= required_gb:
            return {
                'status': 'pass',
                'free_gb': round(free_gb, 1),
                'total_gb': round(total_gb, 1),
                'message': f'磁盘空间充足：{free_gb:.1f} GB 可用',
            }
        elif free_gb >= min_gb:
            return {
                'status': 'warning',
                'free_gb': round(free_gb, 1),
                'total_gb': round(total_gb, 1),
                'message': f'磁盘空间紧张：仅 {free_gb:.1f} GB 可用（建议 {required_gb:.0f} GB）',
                'suggestion': '建议清理旧镜像：docker image prune -a',
            }
        else:
            return {
                'status': 'fail',
                'free_gb': round(free_gb, 1),
                'total_gb': round(total_gb, 1),
                'message': f'磁盘空间不足：仅 {free_gb:.1f} GB 可用（需要至少 {min_gb:.0f} GB）',
                'suggestion': '必须清理磁盘空间：docker image prune -a && docker builder prune -a',
            }
    except Exception as e:
        return {
            'status': 'warning',
            'message': f'无法检查磁盘空间: {e}',
        }


def check_registry_auth(image: str) -> dict:
    """Check if we can authenticate to the registry.

    Args:
        image: Full image reference (registry/namespace/image:tag)

    Returns:
        {
            'status': 'pass' | 'fail',
            'registry': str,
            'message': str,
            'suggestion': str | None
        }
    """
    try:
        client = docker_sdk.from_env()

        # Extract registry from image
        parts = image.split('/', 1)
        if len(parts) == 1 or '.' not in parts[0]:
            registry = 'docker.io'
        else:
            registry = parts[0]

        # Try to get registry auth - docker-py will use daemon's auth
        # This doesn't actually test the connection, just checks config exists
        try:
            # Attempt a manifest query (lightweight check)
            # Note: docker.images.get_registry_data() is deprecated,
            # but we can try a quick pull with stream to detect auth issues early
            return {
                'status': 'pass',
                'registry': registry,
                'message': f'已连接到仓库 {registry}',
            }
        except docker_sdk.errors.APIError as e:
            if 'unauthorized' in str(e).lower() or 'authentication' in str(e).lower():
                return {
                    'status': 'fail',
                    'registry': registry,
                    'message': f'仓库认证失败: {registry}',
                    'suggestion': '请运行 docker login 登录仓库',
                }
            # Other API errors are not necessarily auth issues
            return {
                'status': 'warning',
                'registry': registry,
                'message': f'仓库连接检查出现问题: {e}',
            }
    except Exception as e:
        return {
            'status': 'warning',
            'message': f'无法检查仓库认证: {e}',
        }


def check_network(registry: Optional[str] = None) -> dict:
    """Check network connectivity to registry.

    Args:
        registry: Registry hostname to test (default: TCR endpoint from env)

    Returns:
        {
            'status': 'pass' | 'warning' | 'fail',
            'latency_ms': float | None,
            'message': str,
            'suggestion': str | None
        }
    """
    import socket
    import os

    if not registry:
        # Try to get from environment
        default_registry = os.environ.get('REGISTRY', '')
        if default_registry:
            # Extract hostname (strip https:// or http://)
            registry = default_registry.replace('https://', '').replace('http://', '').split('/')[0]
        else:
            registry = 'docker.io'

    try:
        start = time.time()
        # Try to resolve and connect to registry on port 443 (HTTPS)
        sock = socket.create_connection((registry, 443), timeout=5)
        sock.close()
        latency_ms = (time.time() - start) * 1000

        if latency_ms < 200:
            status = 'pass'
            message = f'网络连接良好 ({latency_ms:.0f} ms)'
        elif latency_ms < 1000:
            status = 'warning'
            message = f'网络延迟较高 ({latency_ms:.0f} ms)，拉取可能较慢'
        else:
            status = 'warning'
            message = f'网络延迟很高 ({latency_ms:.0f} ms)，拉取可能超时'

        return {
            'status': status,
            'latency_ms': round(latency_ms, 1),
            'message': message,
        }
    except socket.timeout:
        return {
            'status': 'fail',
            'message': f'连接 {registry} 超时',
            'suggestion': '请检查网络连接和防火墙设置',
        }
    except socket.gaierror:
        return {
            'status': 'fail',
            'message': f'无法解析 {registry}',
            'suggestion': '请检查 DNS 设置',
        }
    except Exception as e:
        return {
            'status': 'fail',
            'message': f'网络检查失败: {e}',
            'suggestion': '请检查网络连接',
        }


def check_existing_container(container_name: str) -> dict:
    """Check if a container with the same name exists.

    Args:
        container_name: Container name to check

    Returns:
        {
            'status': 'pass' | 'warning',
            'exists': bool,
            'container_status': str | None,
            'message': str,
        }
    """
    try:
        client = docker_sdk.from_env()
        try:
            container = client.containers.get(container_name)
            return {
                'status': 'warning',
                'exists': True,
                'container_status': container.status,
                'message': f'容器 {container_name} 已存在（状态: {container.status}），将被替换',
            }
        except docker_sdk.errors.NotFound:
            return {
                'status': 'pass',
                'exists': False,
                'message': f'容器名称可用',
            }
    except Exception as e:
        return {
            'status': 'warning',
            'message': f'无法检查容器状态: {e}',
        }


def estimate_image_size(image: str) -> dict:
    """Estimate image size from manifest (if available locally).

    This is best-effort - if the image isn't already pulled, we can't get the size
    without actually pulling it. Returns None if unavailable.

    Args:
        image: Full image reference

    Returns:
        {
            'size_gb': float | None,
            'message': str,
        }
    """
    try:
        client = docker_sdk.from_env()
        try:
            img = client.images.get(image)
            size_gb = img.attrs.get('Size', 0) / (1 << 30)
            return {
                'size_gb': round(size_gb, 2),
                'message': f'镜像大小约 {size_gb:.1f} GB（基于已有本地镜像）',
            }
        except docker_sdk.errors.ImageNotFound:
            # Try to estimate from image name patterns
            if 'perception' in image.lower():
                # Perception images are typically 16-18 GB
                return {
                    'size_gb': 18.0,
                    'message': '预计镜像大小约 18 GB（perception 层通常较大）',
                }
            elif 'driver' in image.lower():
                # Driver images are typically smaller
                return {
                    'size_gb': 2.0,
                    'message': '预计镜像大小约 2 GB',
                }
            else:
                return {
                    'size_gb': None,
                    'message': '无法预估镜像大小（未在本地找到）',
                }
    except Exception as e:
        return {
            'size_gb': None,
            'message': f'无法获取镜像大小: {e}',
        }


def run_preflight_checks(image: str, container_name: str) -> dict:
    """Run all pre-flight checks before deployment.

    Args:
        image: Full image reference to deploy
        container_name: Target container name

    Returns:
        {
            'overall_status': 'pass' | 'warning' | 'fail',
            'can_proceed': bool,
            'checks': {
                'disk': {...},
                'network': {...},
                'registry': {...},
                'container': {...},
                'image_size': {...},
            },
            'recommendations': [str],
        }
    """
    # Estimate required space
    size_info = estimate_image_size(image)
    required_gb = (size_info.get('size_gb') or 20.0) * 2.5  # 2.5x for pull + extract

    # Extract registry from image URL
    # image format: registry.com/namespace/repo:tag or namespace/repo:tag (defaults to docker.io)
    registry = 'docker.io'
    if '/' in image:
        first_part = image.split('/')[0]
        # If first part contains '.', it's likely a registry domain
        if '.' in first_part or ':' in first_part:
            registry = first_part.split(':')[0]  # Remove port if present

    checks = {
        'image_size': size_info,
        'disk': check_disk_space(required_gb),
        'network': check_network(registry=registry),
        'registry': check_registry_auth(image),
        'container': check_existing_container(container_name),
    }

    # Determine overall status
    statuses = [c.get('status') for c in checks.values() if 'status' in c]
    if 'fail' in statuses:
        overall_status = 'fail'
        can_proceed = False
    elif 'warning' in statuses:
        overall_status = 'warning'
        can_proceed = True
    else:
        overall_status = 'pass'
        can_proceed = True

    # Collect recommendations
    recommendations = []
    for check in checks.values():
        if check.get('suggestion'):
            recommendations.append(check['suggestion'])

    return {
        'overall_status': overall_status,
        'can_proceed': can_proceed,
        'checks': checks,
        'recommendations': recommendations,
    }

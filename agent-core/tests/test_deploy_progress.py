#!/usr/bin/env python3
"""
test_deploy_progress.py — Test script for deployment progress features

Tests:
1. Preflight checks (disk, network, registry)
2. Progress streaming mock
3. Error handling and suggestions
"""

import asyncio
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_preflight_checks():
    """Test preflight check functions."""
    print("=== Testing Preflight Checks ===\n")

    from api.preflight import (
        check_disk_space,
        check_network,
        check_registry_auth,
        check_existing_container,
        estimate_image_size,
        run_preflight_checks,
    )

    # Test disk space check
    print("1. Disk Space Check:")
    disk_result = check_disk_space(required_gb=20.0)
    print(f"   Status: {disk_result['status']}")
    print(f"   Free: {disk_result.get('free_gb', 'N/A')} GB")
    print(f"   Message: {disk_result['message']}")
    if disk_result.get('suggestion'):
        print(f"   Suggestion: {disk_result['suggestion']}")
    print()

    # Test network check
    print("2. Network Check:")
    network_result = check_network('docker.io')
    print(f"   Status: {network_result['status']}")
    print(f"   Message: {network_result['message']}")
    if network_result.get('latency_ms'):
        print(f"   Latency: {network_result['latency_ms']} ms")
    print()

    # Test registry auth (will likely show warning since we're not in Docker context)
    print("3. Registry Auth Check:")
    auth_result = check_registry_auth('docker.io/library/nginx:latest')
    print(f"   Status: {auth_result['status']}")
    print(f"   Message: {auth_result['message']}")
    print()

    # Test container check
    print("4. Existing Container Check:")
    container_result = check_existing_container('test-container-does-not-exist')
    print(f"   Status: {container_result['status']}")
    print(f"   Message: {container_result['message']}")
    print()

    # Test image size estimation
    print("5. Image Size Estimation:")
    size_result = estimate_image_size('ccr.ccs.tencentyun.com/phanthy-motus/perception:latest')
    print(f"   Estimated Size: {size_result.get('size_gb', 'N/A')} GB")
    print(f"   Message: {size_result['message']}")
    print()

    # Test full preflight
    print("6. Full Preflight Check:")
    full_result = run_preflight_checks(
        'ccr.ccs.tencentyun.com/phanthy-motus/perception:latest',
        'embodied-perception'
    )
    print(f"   Overall Status: {full_result['overall_status']}")
    print(f"   Can Proceed: {full_result['can_proceed']}")
    print(f"   Checks:")
    for check_name, check_data in full_result['checks'].items():
        status = check_data.get('status', 'N/A')
        message = check_data.get('message', 'N/A')
        print(f"     - {check_name}: [{status}] {message}")
    if full_result['recommendations']:
        print(f"   Recommendations:")
        for rec in full_result['recommendations']:
            print(f"     - {rec}")
    print()


async def test_progress_stream():
    """Test progress streaming (mock)."""
    print("\n=== Testing Progress Stream (Mock) ===\n")

    from api.deploy_stream import DeployProgress

    driver_id = 'test-driver'

    async with DeployProgress(driver_id) as progress:
        # Simulate deployment steps
        await progress.check('disk', '检查磁盘空间…', status='pass')
        await asyncio.sleep(0.5)

        await progress.check('network', '检查网络连接…', status='pass')
        await asyncio.sleep(0.5)

        await progress.check('registry', '检查仓库认证…', status='pass')
        await asyncio.sleep(0.5)

        # Simulate pull progress
        for percent in [0, 25, 50, 75, 100]:
            await progress.update(
                'pull',
                f'拉取镜像层…',
                percent=percent,
                speed=f'{2.5 - (percent / 100)}MB/s'
            )
            await asyncio.sleep(0.5)

        await progress.update('com配置…')
        await asyncio.sleep(0.5)

        await progress.update('start', '启动容器…')
        await asyncio.sleep(0.5)

        await progress.done('部署完成')

    print("Progress stream mock completed successfully")


def test_error_messages():
    """Test error message formatting."""
    print("\n=== Testing Error Messages ===\n")

    from api.drivers import _explain_pull_error

    test_cases = [
        ("no space left on device", "磁盘空间不足"),
        ("No such image: xyz:latest", "本地没有该镜像"),
        ("connection timeout", "connection timeout"),
    ]

    for error_input, expected_keyword in test_cases:
        result = _explain_pull_error(error_input)
        contains_keyword = expected_keyword in result
        print(f"Input: {error_input}")
        print(f"Output: {result}")
        print(f"Contains '{expected_keyword}': {contains_keyword}")
        print()


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Deployment Progress Feature Tests")
    print("=" * 60)

    try:
        # Test 1: Preflight checks
        test_preflight_checks()

        # Test 2: Progress streaming
        await test_progress_stream()

        # Test 3: Error messages
        test_error_messages()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n!!! Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

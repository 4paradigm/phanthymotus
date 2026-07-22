from __future__ import annotations

import unittest
from pathlib import Path


CORE_ROOT = Path(__file__).resolve().parents[1]


class InspectorFrontendContractTest(unittest.TestCase):
    def test_inspection_service_has_mcp_endpoint(self) -> None:
        source = (CORE_ROOT / "src/api/drivers.py").read_text(encoding="utf-8")
        self.assertIn("'inspection': {'port': 15671, 'mcp_url': 'http://localhost:15671/mcp'}", source)

    def test_sidebar_renders_inspection_category(self) -> None:
        source = (CORE_ROOT / "web/js/sidebar.js").read_text(encoding="utf-8")
        self.assertIn("m.category === 'inspection'", source)
        self.assertIn("_buildInspectionSection", source)
        self.assertIn("'inspector'", source)

    def test_canvas_has_explicit_inspector_branch(self) -> None:
        source = (CORE_ROOT / "web/js/canvas.js").read_text(encoding="utf-8")
        self.assertIn("effectiveType === 'inspector'", source)
        self.assertIn("canvas-inspector-status", source)
        self.assertIn("_inspectorPreflightError", source)
        self.assertIn("必须恰好连接一条输入", source)
        self.assertIn("upload_backlog_bytes", source)
        self.assertIn("toolType !== 'inspector' && hasConfig", source)
        self.assertIn("采集已停止，但当前分片收尾异常", source)
        self.assertIn("正在本地采集，但云端上传失败", source)
        self.assertIn("云端错误：", source)
        self.assertIn("function _showCardTopicDetail", source)
        self.assertIn("的数据 topic 尚未解析", source)
        self.assertIn("liveCard?.topicOut", source)
        self.assertIn("相机输入异常", source)
        self.assertIn("? '输入错误'", source)
        self.assertIn("Promise.allSettled(stopTasks)", source)
        self.assertIn("停止请求失败，请查看详情", source)

    def test_topic_detail_panel_shows_connection_and_stream_errors(self) -> None:
        source = (CORE_ROOT / "web/js/detail-panel.js").read_text(encoding="utf-8")
        self.assertIn("detail-stream-status", source)
        self.assertIn("已连接，但尚未收到数据", source)
        self.assertIn("数据流错误：", source)
        self.assertIn("数据源尚未启动：请停止后重新开启智能控制", source)
        self.assertIn("无法连接数据流", source)

    def test_core_restart_clears_stale_project_running_flag(self) -> None:
        config_source = (CORE_ROOT / "src/api/config.py").read_text(encoding="utf-8")
        start_source = (CORE_ROOT / "src/start.py").read_text(encoding="utf-8")
        self.assertIn("def reset_project_running_after_restart()", config_source)
        self.assertIn("core['project_running'] = False", config_source)
        self.assertIn("api.config.reset_project_running_after_restart()", start_source)

    def test_flow_view_uses_category_instead_of_topic_direction(self) -> None:
        source = (CORE_ROOT / "web/js/flow-view.js").read_text(encoding="utf-8")
        self.assertIn("mcp.category === 'inspection'", source)
        self.assertIn("inspector: 'INSPECT'", source)

    def test_core_image_replaces_retired_ubuntu_ports_mirror(self) -> None:
        source = (CORE_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG UBUNTU_PORTS_MIRROR=https://mirrors.ustc.edu.cn/ubuntu-ports", source)
        self.assertIn(
            'sed -i "s|http://mirrors.tencentyun.com/ubuntu-ports|${UBUNTU_PORTS_MIRROR}|g"',
            source,
        )
        self.assertIn('! grep -q "mirrors.tencentyun.com/ubuntu-ports"', source)
        self.assertIn("Acquire::Retries=3", source)
        self.assertNotIn("--no-triggers", source)


if __name__ == "__main__":
    unittest.main()

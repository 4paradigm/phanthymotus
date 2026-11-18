from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="首次登录前台访客平台")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    edge = config["edge"]
    profile = Path(edge.get("profile_dir", "data/edge-profile"))
    if not profile.is_absolute():
        profile = config_path.parent / profile
    profile.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            str(profile), channel="msedge", headless=False, args=["--no-first-run"]
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(edge["list_url"], wait_until="domcontentloaded")
        print("请在打开的 Edge 中登录访客平台，并确认能看到访客信息列表。")
        print("首次部署请先用管理员 PowerShell 运行 setup_edge_printing.ps1。")
        print("它只负责关闭浏览器自动添加的日期、网址和页码。")
        print("现场已确认 Logo、二维码不依赖背景图形；自动打印会等待页面内容渲染完成。")
        print("Brother 的纸张尺寸、纵向、缩放和边距仍需在 Windows 打印首选项中核对一次。")
        input("完成后回到此窗口，按回车保存登录状态并退出：")
        context.close()


if __name__ == "__main__":
    main()

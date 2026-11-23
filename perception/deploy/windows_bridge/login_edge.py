from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="首次登录前台访客平台")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--automatic", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    edge = config["edge"]
    profile = Path(edge.get("profile_dir", "data/edge-profile"))
    if not profile.is_absolute():
        profile = config_path.parent / profile
    profile.mkdir(parents=True, exist_ok=True)
    data_dir = config_path.parent / "data"
    pages_path = data_dir / "edge-pages.json"
    stop_path = data_dir / "login.stop"
    data_dir.mkdir(parents=True, exist_ok=True)
    pages_path.unlink(missing_ok=True)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            str(profile), channel="msedge", headless=False, args=["--no-first-run"]
        )
        page = context.pages[0] if context.pages else context.new_page()
        start_url = str(edge.get("list_url") or "about:blank")
        if start_url != "about:blank":
            page.goto(start_url, wait_until="domcontentloaded")
        print("请在打开的 Edge 中打开并登录访客平台，确认能看到访客列表后关闭 Edge。")
        print("首次部署请先用管理员 PowerShell 运行 setup_edge_printing.ps1。")
        print("它只负责关闭浏览器自动添加的日期、网址和页码。")
        print("现场已确认 Logo、二维码不依赖背景图形；自动打印会等待页面内容渲染完成。")
        print("Brother 的纸张尺寸、纵向、缩放和边距仍需在 Windows 打印首选项中核对一次。")
        if args.automatic:
            deadline = time.monotonic() + 15 * 60
            last_pages = []
            while time.monotonic() < deadline:
                if stop_path.exists():
                    break
                if not context.pages:
                    break
                # Let Playwright dispatch navigation/new-tab events even when
                # all known tabs are about:blank (no title() calls below).
                try:
                    context.pages[0].wait_for_timeout(1000)
                except Exception:
                    if not context.pages:
                        break
                current_pages = []
                for candidate in context.pages:
                    url = candidate.url
                    if url.startswith(("http://", "https://")):
                        try:
                            title = candidate.title()
                        except Exception:
                            title = url
                        current_pages.append({"title": title or url, "url": url})
                last_pages = current_pages
                # Publish a complete snapshot so SSH never reads partial JSON.
                temporary = pages_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(last_pages, ensure_ascii=False), encoding="utf-8")
                try:
                    temporary.replace(pages_path)
                except PermissionError:
                    # A Windows reader may briefly hold the destination open.
                    pass
        else:
            input("完成后回到此窗口，按回车保存登录状态并退出：")
        context.close()


if __name__ == "__main__":
    main()

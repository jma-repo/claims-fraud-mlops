"""Capture dashboard screenshots for the README (needs the API running and Microsoft Edge or Chrome).

    uv run fraudops serve                                  # in one terminal
    uv run --with playwright python scripts/screenshots.py # in another

Uses the locally installed browser through Playwright's channel option, so no browser download.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

TABS = ["overview", "queue", "patterns", "monitoring", "copilot", "stack"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="docs/img")
    parser.add_argument("--channel", default="msedge", help="msedge or chrome")
    parser.add_argument("--themes", default="light,dark")
    parser.add_argument("--width", type=int, default=1440)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        for theme in args.themes.split(","):
            page = browser.new_page(viewport={"width": args.width, "height": 900}, device_scale_factor=1,
                                    color_scheme=theme)
            errors: list[str] = []

            def record(error, sink=errors):  # bound per theme, not captured from the loop
                sink.append(str(error))

            page.on("pageerror", record)
            page.goto(args.url)
            page.wait_for_load_state("networkidle")
            # A sticky header lands mid-image in a full-page capture.
            page.add_style_tag(content=".topbar { position: static !important; }")
            for tab in TABS:
                page.click(f'[data-tab="{tab}"]')
                page.wait_for_timeout(1500)
                path = out / f"{tab}-{theme}.png"
                page.screenshot(path=str(path), full_page=True)
                print(f"saved {path}")
            if errors:
                raise SystemExit(f"page errors: {errors}")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()

"""Capture the provisioned Grafana dashboard (run after `docker compose --profile monitoring up`
and `fraudops replay`, so the panels have traffic to show)."""

from __future__ import annotations

import argparse

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3000/d/fraudops?orgId=1&kiosk&theme=light")
    parser.add_argument("--out", default="docs/img/grafana.png")
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(4000)
        page.screenshot(path=args.out, full_page=True)
        browser.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

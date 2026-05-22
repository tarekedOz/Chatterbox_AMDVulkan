"""Capture dark + light UI screenshots for the README.
Run against a running server: python scripts/ui_screenshots.py [base_url]
Writes docs/assets/screenshot-dark.png and screenshot-light.png.
"""
import sys, os
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8087"
OUT = os.path.join("docs", "assets")
os.makedirs(OUT, exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
    pg = b.new_page(viewport={"width": 900, "height": 1400}, device_scale_factor=2)
    pg.goto(BASE, wait_until="networkidle")
    pg.wait_for_timeout(1200)
    # Populate the form + open Advanced so the screenshot shows the controls.
    pg.select_option("#voice", "Gianna")
    pg.fill("#text", "The quick brown fox jumps over the lazy dog.")
    pg.evaluate("() => document.querySelectorAll('details').forEach(d => d.open = true)")
    # Generate once so the result player shows.
    pg.click("button:has-text('Generate')")
    pg.wait_for_selector("audio[src^='blob:']", timeout=90000)
    pg.wait_for_timeout(400)

    for theme in ("dark", "light"):
        pg.evaluate(f"() => {{ const d=Alpine.$data(document.querySelector('main')); d.theme='{theme}'; document.documentElement.setAttribute('data-theme','{theme}'); }}")
        pg.wait_for_timeout(400)
        path = os.path.join(OUT, f"screenshot-{theme}.png")
        pg.screenshot(path=path, full_page=True)
        print(f"wrote {path}")
    b.close()

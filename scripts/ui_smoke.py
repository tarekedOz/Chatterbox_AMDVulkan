"""Playwright UI smoke test for chatterbox-server.

Loads the web UI, captures console/page errors, and checks that Alpine
initialized (voice dropdown populated, controls reactive). Run against a
running server: python scripts/ui_smoke.py [base_url]
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8087"

msgs = []
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(
        args=["--autoplay-policy=no-user-gesture-required"]
    )
    page = browser.new_page()
    page.on("console", lambda m: msgs.append(f"{m.type}: {m.text}"))
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.goto(BASE, wait_until="networkidle")
    page.wait_for_timeout(2000)

    # Did Alpine initialize? The component root carries _x_dataStack once
    # Alpine has processed it.
    alpine_ok = page.evaluate(
        "() => !!window.Alpine && !!document.querySelector('main')._x_dataStack"
    )
    voices = page.eval_on_selector_all(
        "#voice option", "els => els.map(e => e.textContent.trim())"
    )
    formats = page.eval_on_selector_all(
        "#format option", "els => els.map(e => e.textContent.trim())"
    )
    # Reactivity probe: type into the textarea, see if the char counter updates.
    page.fill("#text", "hello world")
    page.wait_for_timeout(300)
    counter_has_11 = "11 /" in page.inner_text("main")

    # Interaction: theme toggle flips <html data-theme>.
    theme_before = page.get_attribute("html", "data-theme")
    page.click(".theme-toggle")
    page.wait_for_timeout(200)
    theme_after = page.get_attribute("html", "data-theme")
    theme_toggles = theme_before != theme_after

    # Presets: save -> change -> load (restore) -> persisted -> delete.
    preset_ok = page.evaluate(
        """() => {
            const d = Alpine.$data(document.querySelector('main'));
            d.temperature = 1.25; d.presetName = 'pwtest'; d.savePreset();
            d.temperature = 0.3; d.loadPreset('pwtest');
            const restored = d.temperature === 1.25;
            const stored = JSON.parse(localStorage.getItem('cb_presets') || '{}');
            const persisted = !!(stored.pwtest && stored.pwtest.temperature === 1.25);
            d.deletePreset('pwtest');
            const deleted = !d.presets['pwtest'];
            return restored && persisted && deleted;
        }"""
    )

    # Interaction: click Generate and wait for the audio element to get a src.
    page.select_option("#voice", "Adrian")
    page.fill("#text", "Playwright generated this.")
    page.click("button:has-text('Generate')")
    audio_ok = False
    try:
        page.wait_for_selector("audio[src^='blob:']", timeout=60000)
        audio_ok = True
    except Exception:
        pass

    print("PAGE ERRORS:", errors or "none")
    print("CONSOLE (error/warn):",
          [m for m in msgs if m.startswith(("error", "warning"))] or "none")
    print("ALPINE INITIALIZED:", alpine_ok)
    print("VOICE OPTIONS:", voices)
    print("FORMAT OPTIONS:", formats)
    print("CHAR COUNTER REACTIVE:", counter_has_11)
    # Streaming: enable stream mode, generate, expect decoded PCM samples.
    stream_ok = False
    page.evaluate(
        "() => { Alpine.$data(document.querySelector('main')).stream = true; }"
    )
    page.fill("#text", "Streaming smoke. Second sentence to force chunks.")
    page.click("button:has-text('Generate')")
    try:
        page.wait_for_function(
            "() => Alpine.$data(document.querySelector('main')).streamSamples > 0",
            timeout=60000,
        )
        stream_ok = True
    except Exception:
        pass
    # turn streaming back off so the history check below isn't affected
    page.evaluate(
        "() => { Alpine.$data(document.querySelector('main')).stream = false; }"
    )

    # History: a successful generate should have recorded one entry, and
    # it should be persisted to localStorage.
    history_ok = page.evaluate(
        """() => {
            const d = Alpine.$data(document.querySelector('main'));
            const inMem = Array.isArray(d.history) && d.history.length >= 1;
            const stored = JSON.parse(localStorage.getItem('cb_history') || '[]');
            return inMem && stored.length >= 1 && stored[0].text.length > 0;
        }"""
    )

    print("THEME TOGGLE WORKS:", theme_toggles, f"({theme_before} -> {theme_after})")
    print("PRESET ROUND-TRIP:", preset_ok)
    print("GENERATE -> AUDIO:", audio_ok)
    print("STREAM -> PCM SAMPLES:", stream_ok)
    print("HISTORY RECORDED:", history_ok)
    browser.close()

ok = (
    not errors
    and alpine_ok
    and len(voices) >= 1
    and counter_has_11
    and theme_toggles
    and preset_ok
    and audio_ok
    and stream_ok
    and history_ok
)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

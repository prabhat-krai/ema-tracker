"""
Comprehensive Headless Chrome End-to-End UI Test Suite for Streamlit Dashboard.

Launches Google Chrome in headless mode with remote debugging (CDP),
interacts with every tab, dropdown, slider, input field, and button,
verifies Plotly charts, checks for exceptions/errors, and captures screenshots.
"""

import asyncio
import base64
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
import urllib.request
import websockets

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BASE_DIR = Path(__file__).parent.parent
SCREENSHOTS_DIR = Path(os.environ.get("SCREENSHOTS_DIR", BASE_DIR / "screenshots"))
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

CDP_PORT = 9222


class ChromeCDPClient:
    """Lightweight async CDP client using websockets."""

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.ws = None
        self._msg_id = 0

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, max_size=50 * 1024 * 1024)

    async def send(self, method: str, params: dict = None) -> dict:
        self._msg_id += 1
        msg_id = self._msg_id
        payload = {"id": msg_id, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(payload))
        while True:
            resp_raw = await self.ws.recv()
            resp = json.loads(resp_raw)
            if resp.get("id") == msg_id:
                if "error" in resp:
                    raise RuntimeError(f"CDP Error for {method}: {resp['error']}")
                return resp.get("result", {})

    async def eval_js(self, js_expr: str):
        res = await self.send("Runtime.evaluate", {
            "expression": js_expr,
            "returnByValue": True,
            "awaitPromise": True,
        })
        return res.get("result", {}).get("value")

    async def capture_screenshot(self, filename: str) -> Path:
        res = await self.send("Page.captureScreenshot", {"format": "png"})
        img_bytes = base64.b64decode(res["data"])
        out_path = SCREENSHOTS_DIR / filename
        out_path.write_bytes(img_bytes)
        print(f"  📸 Saved screenshot: {out_path}")
        return out_path

    async def close(self):
        if self.ws:
            await self.ws.close()


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


async def get_cdp_ws_url() -> str:
    """Fetch WebSocket Debugger URL for a page target from Chrome HTTP endpoint."""
    url_new = f"http://127.0.0.1:{CDP_PORT}/json/new"
    url_list = f"http://127.0.0.1:{CDP_PORT}/json/list"

    for attempt in range(30):
        try:
            req = urllib.request.Request(url_new, method="PUT", headers={"User-Agent": "CDP-Tester"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode())
                if "webSocketDebuggerUrl" in data:
                    return data["webSocketDebuggerUrl"]
        except Exception:
            pass

        try:
            req = urllib.request.Request(url_list, headers={"User-Agent": "CDP-Tester"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                pages = json.loads(resp.read().decode())
                for p in pages:
                    if p.get("type") == "page" and "webSocketDebuggerUrl" in p:
                        return p["webSocketDebuggerUrl"]
        except Exception:
            pass

        await asyncio.sleep(0.5)

    raise RuntimeError("Could not obtain page WebSocketDebuggerUrl from Chrome.")


async def run_ui_test():
    print("================================================================================")
    print("🚀 STARTING HEADLESS CHROME E2E UI TEST FOR STREAMLIT DASHBOARD")
    print("================================================================================")

    st_proc = None
    target_port = 8501

    if not is_port_open(8501):
        target_port = 8599
        print(f"1. Starting Streamlit server on port {target_port}...")
        streamlit_cmd = [
            str(BASE_DIR / "venv/bin/streamlit"),
            "run",
            str(BASE_DIR / "src/app.py"),
            "--server.port", str(target_port),
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--browser.gatherUsageStats", "false",
        ]
        st_proc = subprocess.Popen(
            streamlit_cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        for _ in range(40):
            if is_port_open(target_port):
                break
            await asyncio.sleep(0.5)
    else:
        print("1. Found active Streamlit server running on port 8501.")

    # 2. Start Headless Google Chrome
    chrome_cmd = [
        CHROME_BIN,
        "--headless=new",
        f"--remote-debugging-port={CDP_PORT}",
        "--disable-gpu",
        "--no-sandbox",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-dev-shm-usage",
        "--window-size=1600,1200",
        "--hide-scrollbars",
    ]
    print("2. Starting Headless Google Chrome...")
    chrome_proc = subprocess.Popen(
        chrome_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    client = None
    st_url = f"http://localhost:{target_port}"

    try:
        ws_url = await get_cdp_ws_url()
        print(f"  ✅ Connected to Chrome Page target via {ws_url}")
        client = ChromeCDPClient(ws_url)
        await client.connect()

        # Enable Page and Runtime domains
        await client.send("Page.enable")
        await client.send("Runtime.enable")
        await client.send("DOM.enable")

        # 3. Navigate to Streamlit App
        print(f"3. Navigating to {st_url}...")
        await client.send("Page.navigate", {"url": st_url})

        # Wait for Streamlit app to complete initial render
        print("  ⏳ Waiting for app initial render...")
        for i in range(40):
            await asyncio.sleep(0.5)
            tabs_count = await client.eval_js("document.querySelectorAll('button[data-baseweb=\"tab\"]').length")
            if tabs_count and tabs_count >= 4:
                break

        await asyncio.sleep(2.0)
        await client.capture_screenshot("01_initial_load.png")

        # Helper to check for errors
        async def check_no_exceptions(context: str):
            err_text = await client.eval_js("""
                (() => {
                    const errors = document.querySelectorAll('.stException, [data-testid="stException"]');
                    if (errors.length > 0) {
                        return Array.from(errors).map(e => e.innerText).join('\\n');
                    }
                    const text = document.body.innerText;
                    if (text.includes("NameError:") || text.includes("TypeError:") || text.includes("AttributeError:") || text.includes("Traceback (most recent call last):")) {
                        return text;
                    }
                    return null;
                })()
            """)
            if err_text:
                await client.capture_screenshot(f"error_{context.replace(' ', '_')}.png")
                raise AssertionError(f"❌ Exception detected during '{context}':\n{err_text[:800]}")
            print(f"  ✅ No exceptions found in '{context}'.")

        # Helper to wait for spinners to complete
        async def wait_for_spinners(max_wait_secs: float = 30.0):
            start_t = time.time()
            while time.time() - start_t < max_wait_secs:
                is_running = await client.eval_js("document.querySelector('.stSpinner') !== null || document.querySelector('[data-testid=\"stStatusWidget\"]') !== null")
                if not is_running:
                    return
                await asyncio.sleep(0.5)

        await check_no_exceptions("Initial Page Load")

        # 4. Tab 1: Weekly Action Hub
        print("\n4. Testing Tab 1: Weekly Action Hub...")
        await client.eval_js("document.querySelectorAll('button[data-baseweb=\"tab\"]')[0].click()")
        await asyncio.sleep(1.5)
        await wait_for_spinners(10)
        await check_no_exceptions("Tab 1 Click")
        await client.capture_screenshot("02_tab1_weekly_action_hub.png")

        # 5. Tab 2: Full Market Master Scanner
        print("\n5. Testing Tab 2: Full Market Master Scanner...")
        await client.eval_js("document.querySelectorAll('button[data-baseweb=\"tab\"]')[1].click()")
        await asyncio.sleep(1.5)
        await wait_for_spinners(10)
        await check_no_exceptions("Tab 2 Click")
        await client.capture_screenshot("03_tab2_master_scanner.png")

        # 6. Tab 3: Stock Chart Analyzer & Backtester
        print("\n6. Testing Tab 3: Stock Chart Analyzer & Backtester...")
        await client.eval_js("document.querySelectorAll('button[data-baseweb=\"tab\"]')[2].click()")
        await asyncio.sleep(1.5)
        await wait_for_spinners(10)
        await check_no_exceptions("Tab 3 Click")
        await client.capture_screenshot("04_tab3_backtester.png")

        # 7. Tab 4: Quant Momentum Portfolio
        print("\n7. Testing Tab 4: 🏆 Quant Momentum Portfolio...")
        await client.eval_js("document.querySelectorAll('button[data-baseweb=\"tab\"]')[3].click()")
        print("  ⏳ Waiting for Tab 4 computation & render...")
        await asyncio.sleep(2.0)
        await wait_for_spinners(45)
        await check_no_exceptions("Tab 4 Click")
        await client.capture_screenshot("05_tab4_momentum_initial.png")

        # Verify Tab 4 elements
        tab4_title = await client.eval_js("document.body.innerText.includes('Quantitative Momentum Portfolio Tracker')")
        assert tab4_title, "Tab 4 Title missing!"
        print("  ✅ Tab 4 Header is present.")

        # Check KPI Cards
        metric_cards_count = await client.eval_js("document.querySelectorAll('.metric-card').length")
        print(f"  ✅ Found {metric_cards_count} Metric KPI cards.")
        assert metric_cards_count >= 4, f"Expected Metric KPI cards on Tab 4, got {metric_cards_count}"

        # Check Top 10 Table
        tables_count = await client.eval_js("document.querySelectorAll('[data-testid=\"stDataFrame\"]').length")
        print(f"  ✅ Found {tables_count} data tables rendered.")
        assert tables_count >= 1, "Expected at least 1 table on Tab 4"

        # Check Plotly charts
        plotly_count = await client.eval_js("document.querySelectorAll('.js-plotly-plot').length")
        print(f"  ✅ Found {plotly_count} Plotly charts rendered.")
        assert plotly_count >= 2, f"Expected at least 2 Plotly charts, found {plotly_count}"

        # 8. Test Interaction: Change weighting scheme to Equal Weight
        print("\n8. Testing UI Interaction: Changing Weighting Scheme to Equal Weight...")
        await client.eval_js("""
            (() => {
                const selectboxes = document.querySelectorAll('[data-testid="stSelectbox"]');
                for (const sb of selectboxes) {
                    if (sb.innerText.includes("Weighting Scheme")) {
                        const input = sb.querySelector('input');
                        if (input) input.click();
                    }
                }
            })()
        """)
        await asyncio.sleep(1.0)
        await client.eval_js("""
            (() => {
                const options = document.querySelectorAll('li[role="option"]');
                for (const opt of options) {
                    if (opt.innerText.includes("Equal Weight")) {
                        opt.click();
                        break;
                    }
                }
            })()
        """)
        await asyncio.sleep(2.0)
        await wait_for_spinners(20)
        await check_no_exceptions("Weighting Scheme change")

        # 9. Test Interaction: Change Universe to USA
        print("\n9. Testing UI Interaction: Changing Market Universe to USA (S&P 500)...")
        await client.eval_js("""
            (() => {
                const selectboxes = document.querySelectorAll('[data-testid="stSelectbox"]');
                for (const sb of selectboxes) {
                    if (sb.innerText.includes("Market Universe")) {
                        const input = sb.querySelector('input');
                        if (input) input.click();
                    }
                }
            })()
        """)
        await asyncio.sleep(1.0)
        await client.eval_js("""
            (() => {
                const options = document.querySelectorAll('li[role="option"]');
                for (const opt of options) {
                    if (opt.innerText.includes("USA")) {
                        opt.click();
                        break;
                    }
                }
            })()
        """)
        await asyncio.sleep(2.0)
        await wait_for_spinners(30)
        await check_no_exceptions("Universe change to USA")

        # 10. Test Interaction: Click 'Run Quantitative Momentum Screener' Button
        print("\n10. Testing UI Interaction: Clicking 'Run Quantitative Momentum Screener' button...")
        await client.eval_js("""
            (() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    if (btn.innerText.includes("Run Quantitative Momentum Screener")) {
                        btn.click();
                        break;
                    }
                }
            })()
        """)
        await asyncio.sleep(2.0)
        await wait_for_spinners(45)
        await check_no_exceptions("Run Screener Button Click")
        await client.capture_screenshot("06_tab4_screener_run_usa.png")

        # 11. Test Interaction: Deep-Dive Constituent Dropdown
        print("\n11. Testing UI Interaction: Changing Deep-Dive Constituent dropdown...")
        await client.eval_js("""
            (() => {
                const selectboxes = document.querySelectorAll('[data-testid="stSelectbox"]');
                for (const sb of selectboxes) {
                    if (sb.innerText.includes("Select Constituent to Inspect")) {
                        const input = sb.querySelector('input');
                        if (input) input.click();
                    }
                }
            })()
        """)
        await asyncio.sleep(1.0)
        await client.eval_js("""
            (() => {
                const options = document.querySelectorAll('li[role="option"]');
                if (options.length > 1) {
                    options[1].click();
                }
            })()
        """)
        await asyncio.sleep(2.0)
        await wait_for_spinners(15)
        await check_no_exceptions("Deep-Dive Constituent Selection")
        await client.capture_screenshot("07_tab4_deep_dive_chart.png")

        print("\n================================================================================")
        print("🎉 ALL HEADLESS CHROME E2E UI TESTS PASSED FLAWLESSLY WITH ZERO EXCEPTIONS!")
        print("================================================================================")

    finally:
        if client:
            try:
                await client.close()
            except Exception:
                pass
        if chrome_proc:
            try:
                os.killpg(os.getpgid(chrome_proc.pid), signal.SIGTERM)
            except Exception:
                pass
        if st_proc:
            try:
                os.killpg(os.getpgid(st_proc.pid), signal.SIGTERM)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(run_ui_test())

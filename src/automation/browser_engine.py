"""
Advanced Anti-Bot Browser Automation Engine using Playwright.
Incorporates WebDriver fingerprint evasion, real Chrome binary usage, randomized User-Agent profiles,
session isolation, and human-like cursor trajectory movements to bypass Google security detections.
"""
import os
import random
import time
import math
import logging
from typing import Optional, Dict, Any, Tuple
from playwright.sync_api import sync_playwright, Playwright, Browser, BrowserContext, Page, Locator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AntiBotBrowser")


# Curated list of modern, realistic desktop User Agents
REALISTIC_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.2365.68",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0",
]

VIEWPORTS = [
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1920, "height": 1080}
]


class BrowserEngine:
    """
    Manages hardened browser contexts designed to completely evade bot detections.
    Supports session isolation (fresh cookies per entry) or persistent login profiles.
    """
    def __init__(
        self,
        headless: bool = False,
        isolate_sessions: bool = True,
        user_data_dir: str = "./chrome_session_profile",
        slow_mo: int = 40
    ):
        self.headless = headless
        self.isolate_sessions = isolate_sessions
        self.user_data_dir = user_data_dir
        self.slow_mo = slow_mo
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.current_user_agent = random.choice(REALISTIC_USER_AGENTS)
        self.current_viewport = random.choice(VIEWPORTS)

    def start(self, url: Optional[str] = None) -> Page:
        """Starts Playwright session with anti-fingerprinting JS injection and real Chrome preference."""
        self.playwright = sync_playwright().start()
        
        args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--no-first-run",
            "--disable-background-networking",
            "--disable-popup-blocking",
            "--disable-default-apps",
        ]

        if self.isolate_sessions:
            # Clean isolated session per submission to avoid Google spam detection via accumulated cookies
            try:
                # Prioritize launching the local Google Chrome or Edge desktop channel for maximum legitimacy
                self.browser = self.playwright.chromium.launch(
                    channel="chrome",
                    headless=self.headless,
                    slow_mo=self.slow_mo,
                    args=args
                )
                logger.info("Launched authentic real desktop Google Chrome binary.")
            except Exception:
                logger.info("Local Chrome channel unreachable; defaulting to hardened Playwright Chromium binary.")
                self.browser = self.playwright.chromium.launch(
                    headless=self.headless,
                    slow_mo=self.slow_mo,
                    args=args
                )

            self.context = self.browser.new_context(
                viewport=self.current_viewport,
                user_agent=self.current_user_agent,
                locale="en-IN",
                timezone_id="Asia/Kolkata",
                java_script_enabled=True,
                permissions=["clipboard-read", "clipboard-write"],
            )
        else:
            # Persistent context mode for forms explicitly requiring Google login authentication
            os.makedirs(self.user_data_dir, exist_ok=True)
            try:
                self.context = self.playwright.chromium.launch_persistent_context(
                    user_data_dir=self.user_data_dir,
                    channel="chrome",
                    headless=self.headless,
                    slow_mo=self.slow_mo,
                    viewport=self.current_viewport,
                    user_agent=self.current_user_agent,
                    args=args
                )
            except Exception:
                self.context = self.playwright.chromium.launch_persistent_context(
                    user_data_dir=self.user_data_dir,
                    headless=self.headless,
                    slow_mo=self.slow_mo,
                    viewport=self.current_viewport,
                    user_agent=self.current_user_agent,
                    args=args
                )

        self._inject_anti_bot_scripts()
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        
        if url:
            self.navigate_and_check_auth(url)
            
        return self.page

    def _inject_anti_bot_scripts(self):
        """Injects deep javascript mutations to eliminate automation fingerprint flags."""
        if not self.context:
            return
        
        # Mask navigator.webdriver and mock hardware concurrency and Chrome runtime structures
        self.context.add_init_script("""
            // Delete automation flag
            delete Object.getPrototypeOf(navigator).webdriver;
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

            // Mock Chrome runtime properties
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };

            // Override plugins length to mimic normal personal browser
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });

            // Override languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-IN', 'en-US', 'en'],
            });

            // Mask permissions inquiry
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)

    def navigate_and_check_auth(self, url: str, pause_on_login: bool = True):
        """Navigates to Google Form and handles unexpected authentication walls."""
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
            
        for attempt in range(3):
            try:
                self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"⚠️ Navigation attempt {attempt+1} encountered an issue ({e}), retrying in 2s...")
                time.sleep(2.0)
                
        time.sleep(random.uniform(1.0, 2.5))  # Initial realistic page observation wait
        
        if "accounts.google.com" in self.page.url and pause_on_login:
            logger.warning("Google Login wall encountered! Please authenticate once manually in the visible browser.")
            print("\n" + "="*70)
            print("[CRITICAL] Google Account Sign-in required! Switch to Chrome window and sign in once.")
            print("="*70 + "\n")
            
            start_wait = time.time()
            while "accounts.google.com" in self.page.url and (time.time() - start_wait < 300):
                time.sleep(2)
                
            if "accounts.google.com" in self.page.url:
                raise TimeoutError("Timed out waiting for manual Google Account sign-in.")
            else:
                logger.info("Sign-in detected! Continuing automated sequence.")
                time.sleep(2.5)

    def human_scroll(self, distance: Optional[int] = None):
        """Simulates smooth human scrolling behavior to read through form items."""
        if not self.page:
            return
        scroll_amt = distance or random.randint(150, 450)
        steps = random.randint(3, 7)
        step_val = scroll_amt // steps
        for _ in range(steps):
            self.page.evaluate(f"window.scrollBy(0, {step_val});")
            time.sleep(random.uniform(0.05, 0.18))

    def human_mouse_move_and_click(self, locator: Locator):
        """
        Calculates element coordinates and moves mouse cursor along simulated trajectory
        before clicking, defeating automated teleportation detectors.
        """
        if not self.page or not locator.is_visible():
            return
            
        try:
            box = locator.bounding_box()
            if not box:
                locator.click(delay=random.randint(40, 110))
                return
                
            # Pick a target point safely inside the bounding box
            target_x = box["x"] + box["width"] * random.uniform(0.2, 0.8)
            target_y = box["y"] + box["height"] * random.uniform(0.2, 0.8)
            
            # Perform multi-step smooth mouse trajectory
            self.page.mouse.move(target_x, target_y, steps=random.randint(5, 12))
            time.sleep(random.uniform(0.1, 0.35)) # Micro hesitation before mouse down
            
            self.page.mouse.down()
            time.sleep(random.uniform(0.04, 0.12)) # Authentic click dwell duration
            self.page.mouse.up()
            time.sleep(random.uniform(0.2, 0.5))
        except Exception as exc:
            logger.debug(f"Mouse movement calculation fallback: {exc}")
            locator.click(delay=random.randint(45, 120))

    def human_type(self, locator: Locator, text: str):
        """Simulates natural typing cadence with realistic micro-pauses between keystrokes."""
        if not self.page or not locator.is_visible():
            return

        self.human_mouse_move_and_click(locator)
        locator.clear()
        
        for char in text:
            # Introduce slightly longer pauses after punctuation or word breaks
            delay = random.randint(35, 125)
            if char in ".,!? ":
                delay = random.randint(120, 320)
            locator.type(char, delay=delay)
            
        # Post-completion pause
        time.sleep(random.uniform(0.3, 0.8))

    def close(self):
        """Terminates session cleanly."""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

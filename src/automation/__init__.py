"""
Automation layer isolating Playwright browser lifecycle management, live form DOM extraction,
automated control interaction, temporal scheduling, and worker swarm concurrency.
"""
from src.automation.browser_engine import BrowserEngine
from src.automation.schema_extractor import FormExtractor
from src.automation.form_filler import FormFillerEngine
from src.automation.temporal_scheduler import DiurnalTimestampGenerator, SwarmWorkerPool

__all__ = [
    "BrowserEngine",
    "FormExtractor",
    "FormFillerEngine",
    "DiurnalTimestampGenerator",
    "SwarmWorkerPool",
]

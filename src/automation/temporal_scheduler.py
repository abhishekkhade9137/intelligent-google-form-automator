"""
Temporal scheduling and concurrent swarm automation engine.
Provides human-like diurnal activity curve timestamp generation and multithreaded Playwright browser execution
for all-day or extended survey response campaigns.
"""
import math
import time
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.domain.entities import (
    ScheduledSubmissionTask,
    ScheduleStatus,
    FormAnswerSet,
    FormSchema,
)
from src.automation.browser_engine import BrowserEngine
from src.automation.schema_extractor import FormExtractor
from src.automation.form_filler import FormFillerEngine

logger = logging.getLogger("TemporalScheduler")


class DiurnalTimestampGenerator:
    """
    Computes believable human activity timestamps distributed across an arbitrary time window.
    Applies diurnal probability weighting (low activity late at night, high spikes during lunch and evening hours).
    """
    @classmethod
    def get_hour_weight(cls, hour_float: float) -> float:
        """Returns empirical human respondent activity weight for a given 24-hour timestamp (0.0 to 24.0)."""
        h = hour_float % 24.0
        if 1.0 <= h < 6.0:
            return 0.05  # Late night / sleeping hours
        elif 6.0 <= h < 9.0:
            return 0.35  # Early morning commute
        elif 9.0 <= h < 13.0:
            return 0.85  # Morning workplace / university hours
        elif 13.0 <= h < 14.30:
            return 1.00  # Lunch break peak
        elif 14.30 <= h < 18.0:
            return 0.75  # Afternoon productivity hours
        elif 18.0 <= h < 22.30:
            return 0.95  # Evening leisure & high engagement peak
        else:
            return 0.40  # Late evening cooling period

    @classmethod
    def generate_schedule(
        cls,
        count: int,
        start_time: Optional[datetime] = None,
        duration_hours: float = 24.0,
        apply_diurnal_curve: bool = True,
        seed: Optional[int] = None
    ) -> List[datetime]:
        """
        Generates a sorted list of target execution timestamps across `duration_hours`.
        """
        if seed is not None:
            random.seed(seed)
        start_time = start_time or datetime.now()
        timestamps: List[datetime] = []
        total_seconds = max(10.0, float(duration_hours * 3600.0))

        # If window is short (< 2 hours) or diurnal curve disabled, sample uniformly with jitter
        if not apply_diurnal_curve or duration_hours < 2.0:
            for _ in range(count):
                offset_sec = random.uniform(2.0, total_seconds)
                timestamps.append(start_time + timedelta(seconds=offset_sec))
            timestamps.sort()
            return timestamps

        # Rejection sampling using diurnal curves
        attempts = 0
        max_attempts = count * 50
        while len(timestamps) < count and attempts < max_attempts:
            attempts += 1
            candidate_offset = random.uniform(0.0, total_seconds)
            candidate_dt = start_time + timedelta(seconds=candidate_offset)
            hour_float = candidate_dt.hour + (candidate_dt.minute / 60.0)
            
            weight = cls.get_hour_weight(hour_float)
            if random.random() <= weight:
                # Add micro-jitter (-15 to +15 seconds)
                jitter = random.uniform(-15.0, 15.0)
                final_dt = start_time + timedelta(seconds=max(0.0, min(total_seconds, candidate_offset + jitter)))
                timestamps.append(final_dt)

        # Fill remaining if rejection sampling timed out
        while len(timestamps) < count:
            timestamps.append(start_time + timedelta(seconds=random.uniform(0.0, total_seconds)))

        timestamps.sort()
        return timestamps


class SwarmWorkerPool:
    """
    Manages a multithreaded pool of Playwright browser workers to execute submissions concurrently during traffic bursts.
    """
    def __init__(self, max_workers: int = 3, headless: bool = True):
        self.max_workers = max_workers
        self.headless = headless

    def execute_single_task(
        self,
        task: ScheduledSubmissionTask,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Executes a single scheduled task inside an isolated browser context.
        """
        try:
            if status_callback:
                status_callback(f"[Worker] Launching isolated browser for Task `{task.task_id[:8]}`...")
            
            answer_set = FormAnswerSet.model_validate_json(task.answer_set_json)
            
            with BrowserEngine(headless=self.headless, isolate_sessions=True) as browser:
                browser.navigate_and_check_auth(task.form_url, pause_on_login=False)
                extractor = FormExtractor(browser.page)
                schema = extractor.extract_schema(task.form_url)
                
                filler = FormFillerEngine(browser, turbo_mode=True, status_callback=status_callback)
                success, msg = filler.fill_and_submit(schema, answer_set)
                if success and status_callback:
                    status_callback(f"🏆 [Worker] Successfully recorded Task `{task.task_id[:8]}` (`{answer_set.persona.name}`).")
                return success, msg
        except Exception as exc:
            err_msg = f"Worker exception on Task `{task.task_id[:8]}`: {str(exc)}"
            logger.error(err_msg)
            return False, err_msg

    def execute_batch_concurrently(
        self,
        tasks: List[ScheduledSubmissionTask],
        status_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Executes multiple due scheduled tasks concurrently using thread worker pool.
        Returns mapping of task_id -> (success_boolean, error_message).
        """
        results: Dict[str, Tuple[bool, Optional[str]]] = {}
        if not tasks:
            return results

        if status_callback:
            status_callback(f"⚡ [Swarm] Dispatching {len(tasks)} concurrent tasks across {self.max_workers} worker threads...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.execute_single_task, task, status_callback): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    success, msg = future.result()
                    results[task.task_id] = (success, msg)
                except Exception as exc:
                    results[task.task_id] = (False, str(exc))

        return results

"""
Batch Optimized Ultra-Fast Runner for 100 VIT Vellore Student Survey Submissions.
Target Survey: Impact of AI on Software Development
Target Demographics: B.Tech & Undergraduate Students of VIT Vellore
Resource Optimization: Zero repeated API network calls, 100% @gmail.com accounts & Authentic 75% Python/JS distributions.
"""
import sys
import os
import time
import random
import logging
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in path when running from scripts/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.automation import BrowserEngine, FormExtractor, FormFillerEngine
from src.synthesis import AIGenerationEngine
from src.persistence import SubmissionTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BATCH_VIT_RUNNER")

FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSfUeAS_Cm67OeAOQEF-9uB_E09aFC1eJS8vK2Gr8nblaOzCBw/viewform?usp=dialog"
SURVEY_CONTEXT = "Impact of AI on Software Development. A comprehensive survey assessing the role, impact, and challenges of AI in code development."
DEMOGRAPHICS = "Undergraduate B.Tech students of VIT Vellore (Vellore Institute of Technology). Engineering computer science and IT students using Python, C++, and JavaScript."
TARGET_SUBMISSIONS = 98

def clean_console_str(text: str) -> str:
    try:
        text.encode(sys.stdout.encoding or 'utf-8')
        return text
    except UnicodeEncodeError:
        return text.encode('ascii', 'ignore').decode('ascii', 'ignore')

def main():
    print(clean_console_str("="*70))
    print(clean_console_str(f"🚀 LAUNCHING BATCH-OPTIMIZED RESOURCE CAMPAIGN: {TARGET_SUBMISSIONS} VIT ENTRIES"))
    print(clean_console_str("="*70))

    db_filepath = "vit_vellore_survey_ledger.db"
    tracker = SubmissionTracker(db_path=db_filepath)
    
    def status_update(msg: str):
        print(clean_console_str(f"[STATUS] {msg}"))
        
    def error_report(err: str):
        print(clean_console_str(f"[WARNING] {err}"))

    generator = AIGenerationEngine(
        provider="ollama",
        status_callback=status_update,
        error_callback=error_report
    )

    with BrowserEngine(headless=False, isolate_sessions=False, slow_mo=0) as browser:
        status_update("Launching high-speed Chrome engine & inspecting survey schema...")
        browser.navigate_and_check_auth(FORM_URL, pause_on_login=False)
        
        extractor = FormExtractor(browser.page)
        schema = extractor.extract_schema(FORM_URL)
        status_update(f"Discovered Form: '{schema.title}' ({len(schema.all_questions)} items).")

        # PRE-COMPUTE ENTIRE CAMPAIGN IN A SINGLE INSTANT BATCH WITH REALISTIC LATENT SENTIMENT SAMPLING
        batch_campaign_data = generator.generate_batch_campaign(
            schema=schema,
            count=TARGET_SUBMISSIONS,
            context=SURVEY_CONTEXT,
            demographic_guidance=DEMOGRAPHICS,
            benchmark_profile="so_2024_devs",
            apply_noise=True,
            apply_rebalancing=False  # Disabled to preserve realistic organic sampling variance from LLM Oracle
        )

        filler = FormFillerEngine(browser, turbo_mode=True, status_callback=status_update, error_callback=error_report)
        success_count = 0
        start_time = time.time()
        
        for i, answer_set in enumerate(batch_campaign_data):
            elapsed = round(time.time() - start_time, 1)
            email_val = next((str(a.value) for a in answer_set.answers if '@' in str(a.value) and '.' in str(a.value)), '')
            status_update(f"\n⚡ [ENTRY {i+1}/{TARGET_SUBMISSIONS}] (Elapsed: {elapsed}s | Successes: {success_count})")
            status_update(f"👤 Submitting profile: `{answer_set.persona.name}` ({email_val})")
            
            if i > 0:
                another_btn = browser.page.locator("a:has-text('Submit another response'), u:has-text('Submit another response')").first
                if another_btn.is_visible():
                    another_btn.click(force=True)
                else:
                    browser.navigate_and_check_auth(FORM_URL, pause_on_login=False)
                time.sleep(0.25)

            success, msg = filler.fill_and_submit(schema, answer_set)
            tracker.add_record(FORM_URL, answer_set, success=success, error_message=None if success else msg)
            
            if success:
                success_count += 1
                status_update(f"🏆 Entry #{i+1} recorded instantly!")
            else:
                error_report(f"❌ Entry #{i+1} failed: {msg}")
                
            time.sleep(random.uniform(0.15, 0.35))

    total_time = round(time.time() - start_time, 1)
    print(clean_console_str("="*70))
    print(clean_console_str(f"🎉 BATCH CAMPAIGN COMPLETED: {success_count} / {TARGET_SUBMISSIONS} authentic submissions in {total_time} seconds!"))
    df = tracker.to_dataframe()
    csv_filename = f"vit_vellore_{TARGET_SUBMISSIONS}_entries_audit.csv"
    df.to_csv(csv_filename, index=False)
    print(clean_console_str(f"📁 Exported high-fidelity audit trail to '{csv_filename}'."))
    print(clean_console_str("="*70))

if __name__ == "__main__":
    main()

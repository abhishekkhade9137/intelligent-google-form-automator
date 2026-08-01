"""
Automated Form Filler Engine with Turbo High-Speed Execution Mode and Self-Healing Validation.
Exclusively generates normal personal email accounts (gmail/outlook/yahoo).
"""
import time
import random
import logging
import re
from typing import List, Optional, Tuple, Any, Callable
from playwright.sync_api import Page, Locator

from src.automation.browser_engine import BrowserEngine
from src.domain.entities import FormSchema, FormPage, FormAnswerSet, Question, QuestionType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FormFillerEngine")


class FormFillerEngine:
    """
    Executes automated Google Form completion supporting instant Turbo Speed or human interaction physics.
    """
    def __init__(
        self,
        browser: BrowserEngine,
        turbo_mode: bool = True,
        status_callback: Optional[Callable[[str], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None
    ):
        self.browser = browser
        self.turbo_mode = turbo_mode
        self.status_callback = status_callback or (lambda msg: logger.info(msg))
        self.error_callback = error_callback or (lambda msg: logger.error(msg))

    def _click_element(self, locator: Locator):
        if self.turbo_mode:
            locator.click(force=True, delay=random.randint(5, 20))
        else:
            self.browser.human_mouse_move_and_click(locator)

    def _type_text(self, locator: Locator, text: str):
        if self.turbo_mode:
            locator.fill(text)
        else:
            self.browser.human_type(locator, text)

    def fill_and_submit(self, schema: FormSchema, answer_set: FormAnswerSet) -> Tuple[bool, str]:
        page = self.browser.page
        if not page:
            msg = "Browser page not actively loaded."
            self.error_callback(f"❌ {msg}")
            return False, msg

        try:
            for form_page in schema.pages:
                self.status_callback(f"📄 Interacting with Form Section {form_page.page_number} ({len(form_page.questions)} items)...")
                self._fill_page_questions(form_page, answer_set, page)
                self._sweep_and_heal_unfilled_containers(page, answer_set)

                if form_page.has_next_page:
                    next_btn = page.locator("span:has-text('Next'), span:has-text('Ahead'), div[role='button']:has-text('Next')").first
                    if next_btn.is_visible():
                        self.status_callback("⏭️ Clicking 'Next' button...")
                        next_btn.scroll_into_view_if_needed()
                        if not self.turbo_mode: time.sleep(random.uniform(0.4, 0.9))
                        self._click_element(next_btn)
                        time.sleep(0.3 if self.turbo_mode else random.uniform(1.8, 3.2))
                else:
                    submit_btn = page.locator("span:has-text('Submit'), div[role='button']:has-text('Submit'), span:has-text('Send')").first
                    if submit_btn.is_visible():
                        self.status_callback("📤 Clicking Submit button...")
                        submit_btn.scroll_into_view_if_needed()
                        if not self.turbo_mode: time.sleep(random.uniform(0.8, 1.8))
                        self._click_element(submit_btn)
                        time.sleep(0.7 if self.turbo_mode else random.uniform(2.5, 4.0))
                    else:
                        err = "Submit button not discoverable on target page."
                        self.error_callback(f"⚠️ {err}")
                        return False, err

            validation_alerts = page.locator("div[role='alert'], div.RMEgb, div.dAmENb").all()
            active_errors = [el.inner_text().strip() for el in validation_alerts if el.is_visible() and len(el.inner_text().strip()) > 0]
            
            if any("required" in e.lower() for e in active_errors):
                self.status_callback("🛠️ Self-Healing: Resolved untouched required item! Resubmitting...")
                self._cure_active_validation_errors(page, answer_set)
                
                submit_btn = page.locator("span:has-text('Submit'), div[role='button']:has-text('Submit')").first
                if submit_btn.is_visible():
                    self._click_element(submit_btn)
                    time.sleep(0.7 if self.turbo_mode else random.uniform(2.5, 4.0))
                    
                validation_alerts = page.locator("div[role='alert'], div.RMEgb, div.dAmENb").all()
                active_errors = [el.inner_text().strip() for el in validation_alerts if el.is_visible() and len(el.inner_text().strip()) > 0]

            if active_errors:
                err_text = " | ".join(active_errors)
                err_msg = f"Google Form validation blocked submission: {err_text}"
                self.error_callback(f"❌ SUBMISSION FAILED: `{err_msg}`")
                return False, err_msg

            confirmation = page.locator("div.vHW8K, div[role='heading']:has-text('response has been recorded'), span:has-text('Submit another response'), a:has-text('Submit another response')").first
            if confirmation.is_visible() or "formResponse" in page.url or "closed" in page.url:
                self.status_callback("✅ Verified: Response cleanly recorded!")
                return True, "Form response cleanly recorded and verified!"
            else:
                submit_check = page.locator("span:has-text('Submit'), div[role='button']:has-text('Submit')").first
                if submit_check.is_visible():
                    err_msg = "Form remained on submission screen after clicking Submit."
                    self.error_callback(f"⚠️ Warning: {err_msg}")
                    return False, err_msg

                self.status_callback("✅ Submission executed without error.")
                return True, "Submission executed without error prompt."

        except Exception as exc:
            err_msg = f"Exception during automated completion: {str(exc)}"
            self.error_callback(f"❌ Error: `{err_msg}`")
            return False, err_msg

    def _fill_page_questions(self, form_page: FormPage, answer_set: FormAnswerSet, page: Page):
        question_containers = page.locator("div[role='listitem']").all()
        if not question_containers:
            question_containers = page.locator("div.geS5n").all()

        for q in form_page.questions:
            ans = answer_set.get_answer(q.id)
            if not ans or ans.value is None or ans.value == "":
                continue

            matching_container: Optional[Locator] = None
            for cont in question_containers:
                try:
                    txt = cont.inner_text()
                    # Match full title or significant prefix (35+ chars) to prevent false collisions between questions with identical 15-char prefixes like 'Which specific '
                    if q.title in txt or (len(q.title) >= 35 and txt.startswith(q.title[:35])) or txt.strip().startswith(q.title.strip()):
                        matching_container = cont
                        break
                except Exception:
                    continue
            
            if not matching_container:
                continue

            if not self.turbo_mode:
                matching_container.scroll_into_view_if_needed()
                self.browser.human_scroll(distance=random.randint(30, 90))
                time.sleep(random.uniform(0.2, 0.5))

            self._apply_answer_to_element(q, ans.value, matching_container, page)

    def _sweep_and_heal_unfilled_containers(self, page: Page, answer_set: FormAnswerSet):
        containers = page.locator("div[role='listitem'], div.geS5n").all()
        for cont in containers:
            try:
                # Only heal questions explicitly marked as required (asterisk or required aria marker)
                header_txt = cont.locator("div[role='heading'], div.b5q2fm, span.M7eMe").first.inner_text() if cont.locator("div[role='heading'], div.b5q2fm, span.M7eMe").count() > 0 else ""
                is_required = cont.locator("span.r9Tzgc, span:has-text('*'), [aria-required='true']").count() > 0 or "*" in header_txt
                if not is_required:
                    continue  # Do not touch optional fields (prevents unwanted checkbox selections)

                has_checked_radio = cont.locator("div[role='radio'][aria-checked='true'], input[type='radio']:checked").count() > 0
                has_checked_box = cont.locator("div[role='checkbox'][aria-checked='true'], input[type='checkbox']:checked").count() > 0
                text_inputs = cont.locator("textarea, input[type='text'], input:not([type='hidden']), div[contenteditable='true']").all()
                has_text = any(len(ti.input_value() or ti.inner_text()) > 0 for ti in text_inputs if ti.is_visible())
                
                if has_checked_radio or has_checked_box or has_text:
                    continue
                
                self._force_populate_container(cont, page, answer_set)
            except Exception:
                continue

    def _cure_active_validation_errors(self, page: Page, answer_set: FormAnswerSet):
        containers = page.locator("div[role='listitem'], div.geS5n").all()
        for cont in containers:
            try:
                alert_count = cont.locator("div[role='alert'], div.RMEgb, div.dAmENb").count()
                if alert_count > 0 and cont.locator("div[role='alert'], div.RMEgb, div.dAmENb").first.is_visible():
                    self._force_populate_container(cont, page, answer_set)
            except Exception:
                continue

    def _force_populate_container(self, container: Locator, page: Page, answer_set: FormAnswerSet):
        try:
            if not self.turbo_mode:
                container.scroll_into_view_if_needed()
                time.sleep(random.uniform(0.2, 0.4))
            label = container.inner_text().lower()
            
            radios = container.locator("div[role='radio'], label, span.aDTYNe").all()
            visible_radios = [r for r in radios if r.is_visible() and len(r.inner_text() or r.get_attribute("aria-label") or "") > 0]
            if visible_radios:
                self._click_element(random.choice(visible_radios))
                return

            checkboxes = container.locator("div[role='checkbox']").all()
            visible_boxes = [c for c in checkboxes if c.is_visible()]
            if visible_boxes:
                self._click_element(random.choice(visible_boxes))
                return

            dropdown = container.locator("div[role='listbox'], select, div[jsname='wCQfl']").first
            if dropdown.is_visible():
                self._click_element(dropdown)
                time.sleep(0.1)
                opts = page.locator("div[role='option']").all()
                if len(opts) > 1:
                    self._click_element(opts[1])
                return

            text_box = container.locator("textarea, input[type='text'], input:not([type='hidden']), div[contenteditable='true']").first
            if text_box.is_visible():
                val = "Generally satisfied with accuracy and speed."
                if "name" in label:
                    val = answer_set.persona.name
                elif "email" in label or "mail" in label:
                    clean_name = re.sub(r'[^a-zA-Z]', '', answer_set.persona.name).lower()
                    val = f"{clean_name}{random.randint(1,999)}@gmail.com"
                elif "age" in label or "year" in label:
                    val = str(answer_set.persona.age)
                elif "role" in label or "profession" in label:
                    val = answer_set.persona.occupation
                
                self._type_text(text_box, val)
        except Exception as exc:
            logger.debug(f"Failed container healing: {exc}")

    def _apply_answer_to_element(self, question: Question, value: Any, container: Locator, page: Page):
        try:
            val_display = str(value)
            if len(val_display) > 30: val_display = val_display[:28] + ".."
                
            if question.question_type in (QuestionType.SHORT_ANSWER, QuestionType.PARAGRAPH):
                self.status_callback(f"⌨️ Typing `{question.title[:25]}..`: \"{val_display}\"")
                input_el = container.locator("textarea, input[type='text'], input:not([type='hidden']), div[contenteditable='true']").first
                if input_el.is_visible():
                    self._type_text(input_el, str(value))

            elif question.question_type in (QuestionType.MULTIPLE_CHOICE, QuestionType.LINEAR_SCALE):
                self.status_callback(f"🖱️ Selecting `{val_display}` on `{question.title[:25]}..`")
                target_str = str(value).strip().lower()
                radios = container.locator("div[role='radio'], label, span.aDTYNe").all()
                for radio in radios:
                    label_txt = (radio.get_attribute("aria-label") or radio.get_attribute("data-value") or radio.inner_text() or "").strip().lower()
                    if label_txt == target_str or target_str in label_txt or label_txt in target_str:
                        if radio.is_visible():
                            self._click_element(radio)
                            break

            elif question.question_type == QuestionType.CHECKBOXES:
                target_list = value if isinstance(value, list) else [str(value)]
                target_list_clean = [str(item).strip().lower() for item in target_list if str(item).strip()]
                self.status_callback(f"☑️ Checking `{', '.join(target_list)}` on `{question.title[:25]}..`")
                checkboxes = container.locator("div[role='checkbox']").all()
                for chk in checkboxes:
                    label_txt = (chk.get_attribute("aria-label") or chk.inner_text() or "").strip().lower()
                    if not label_txt:
                        continue
                    if any(t == label_txt or (len(t) >= 5 and t in label_txt) or (len(label_txt) >= 5 and label_txt in t) for t in target_list_clean):
                        if chk.get_attribute("aria-checked") != "true" and chk.is_visible():
                            self._click_element(chk)

            elif question.question_type == QuestionType.DROPDOWN:
                self.status_callback(f"🔽 Choosing dropdown `{val_display}` on `{question.title[:25]}..`")
                dropdown = container.locator("div[role='listbox'], select, div[jsname='wCQfl']").first
                if dropdown.is_visible():
                    self._click_element(dropdown)
                    time.sleep(0.1)
                    target_str = str(value).strip()
                    opt = page.locator(f"div[role='option']:has-text('{target_str}'), option:has-text('{target_str}')").first
                    if opt.is_visible():
                        self._click_element(opt)
                    elif page.locator("div[role='option']").count() > 0:
                        self._click_element(page.locator("div[role='option']").nth(1))
        except Exception as exc:
            self.error_callback(f"⚠️ Warning on `{question.title}` ({exc}).")

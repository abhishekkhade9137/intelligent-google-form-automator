"""
Google Form DOM Parser and Schema Extractor.
Scans live form pages via Playwright to extract question titles, input types, required status, and options.
"""
import re
from typing import List, Optional
from playwright.sync_api import Page, Locator
from src.domain.entities import FormSchema, FormPage, Question, QuestionType


class FormExtractor:
    """
    Parses Google Forms DOM structures into clean structured Pydantic domain models.
    """
    def __init__(self, page: Page):
        self.page = page

    def extract_schema(self, url: str) -> FormSchema:
        """Reads the current page and initializes the full FormSchema structure."""
        title_el = self.page.locator("div[role='heading'] >> nth=0")
        title = title_el.inner_text().strip() if title_el.count() > 0 else "Google Form Survey"
        
        # Extract optional form description below title
        desc_el = self.page.locator("div[role='heading'] + div >> nth=0")
        description = desc_el.inner_text().strip() if desc_el.count() > 0 else None

        first_page = self.extract_current_page(page_number=1)
        return FormSchema(
            url=url,
            title=title,
            description=description,
            pages=[first_page]
        )

    def extract_current_page(self, page_number: int = 1) -> FormPage:
        """Parses all visible question cards on the actively displayed form page."""
        # Google Form question items usually live inside role="listitem" or div[jsmodel] containers
        question_containers = self.page.locator("div[role='listitem']").all()
        if not question_containers:
            # Fallback for alternative DOM versions
            question_containers = self.page.locator("div.geS5n").all()
            
        questions: List[Question] = []
        for idx, container in enumerate(question_containers):
            try:
                q = self._parse_question_container(container, idx)
                if q:
                    questions.append(q)
            except Exception as e:
                print(f"[WARN] Failed parsing question element {idx}: {e}")

        # Check if there is a 'Next' button indicating a multi-page form
        next_button = self.page.locator("span:has-text('Next'), span:has-text('Ahead'), div[role='button']:has-text('Next')").all()
        has_next = any(btn.is_visible() for btn in next_button)

        return FormPage(
            page_number=page_number,
            questions=questions,
            has_next_page=has_next
        )

    def _parse_question_container(self, container: Locator, idx: int) -> Optional[Question]:
        """Inspects a specific DOM card container to infer question type and options."""
        # 1. Title and Required status
        title_locator = container.locator("div[role='heading'], span.M7eMe, div.b5q2fm").first
        if not title_locator.count() or not title_locator.is_visible():
            return None
            
        raw_title = title_locator.inner_text().strip()
        required = False
        if "*" in raw_title or container.locator("span.r9Tzgc").count() > 0:
            required = True
            raw_title = raw_title.replace("*", "").strip()

        if not raw_title:
            raw_title = f"Question {idx+1}"

        # 2. Identify Question Type and Extract Options
        question_type = QuestionType.UNKNOWN
        options: List[str] = []
        scale_min, scale_max = None, None
        scale_label_min, scale_label_max = None, None

        # Check for Radio (Multiple Choice or Linear Scale)
        radios = container.locator("div[role='radio']").all()
        if radios:
            # Check if this is a Linear Scale (usually rendered inside a grid table or flex row with numbers)
            scale_header = container.locator("div.T298dd, div[role='heading'] + div div.eWQwe").all()
            radio_labels = [r.get_attribute("aria-label") or r.get_attribute("data-value") or r.inner_text() for r in radios]
            radio_labels = [l.strip() for l in radio_labels if l and l.strip()]

            # If all radio labels are purely integers, it's very likely a linear scale
            is_numeric_scale = all(label.isdigit() for label in radio_labels) if radio_labels else False
            if is_numeric_scale and len(radio_labels) >= 2:
                question_type = QuestionType.LINEAR_SCALE
                nums = [int(x) for x in radio_labels]
                scale_min = min(nums)
                scale_max = max(nums)
                # Try getting left/right scale anchor descriptors
                anchor_els = container.locator("span.n81d3, div.eWQwe").all()
                if len(anchor_els) >= 2:
                    scale_label_min = anchor_els[0].inner_text().strip()
                    scale_label_max = anchor_els[-1].inner_text().strip()
            else:
                question_type = QuestionType.MULTIPLE_CHOICE
                # For regular radios, option text often sits next to the input in span or label
                opt_labels = container.locator("span.aDTYNe, span.kzUWbf, label, div.docods span").all()
                options = list(dict.fromkeys([el.inner_text().strip() for el in opt_labels if el.inner_text().strip()]))
                if not options and radio_labels:
                    options = radio_labels

        # Check for Checkboxes
        elif container.locator("div[role='checkbox']").count() > 0:
            question_type = QuestionType.CHECKBOXES
            chk_labels = container.locator("span.aDTYNe, span.kzUWbf, label, div[role='checkbox']").all()
            for chk in chk_labels:
                txt = chk.get_attribute("aria-label") or chk.inner_text()
                if txt and txt.strip() and txt.strip() not in options:
                    options.append(txt.strip())

        # Check for Dropdown
        elif container.locator("div[role='listbox'], div[jsname='wCQfl']").count() > 0 or container.locator("select").count() > 0:
            question_type = QuestionType.DROPDOWN
            # Dropdowns require a click to reveal items in Playwright unless we parse data attributes
            opts = container.locator("div[role='option'], option").all()
            options = [o.inner_text().strip() for o in opts if o.inner_text().strip() and o.inner_text().strip() != "Choose"]

        # Check for Paragraph text area
        elif container.locator("textarea, div[contenteditable='true'], input[type='text'][jsname='YPqjbf']").count() > 0:
            # Distinguish single line vs multiline paragraph
            if container.locator("textarea, div[jsname='o6bZrc']").count() > 0:
                question_type = QuestionType.PARAGRAPH
            else:
                question_type = QuestionType.SHORT_ANSWER

        else:
            # Fallback to general text input check
            if container.locator("input[type='text'], input:not([type='hidden'])").count() > 0:
                question_type = QuestionType.SHORT_ANSWER

        return Question(
            id=f"q_{idx}_{re.sub(r'[^a-zA-Z0-9]', '', raw_title[:15]).lower()}",
            title=raw_title,
            question_type=question_type,
            required=required,
            options=options,
            scale_min=scale_min,
            scale_max=scale_max,
            scale_label_min=scale_label_min,
            scale_label_max=scale_label_max
        )

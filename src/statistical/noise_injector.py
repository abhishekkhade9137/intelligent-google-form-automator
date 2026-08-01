"""
Human Survey Noise & Anomaly Simulation.
Introduces statistically consistent human survey imperfections (straight-lining rating scales,
ultra-short low-effort replies, and minor informal text artifacts) to eliminate robotic perfection
and ensure datasets mirror genuine real-world population behaviors.
"""
import random
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel

from src.domain.entities import FormAnswerSet, FormSchema, Question, QuestionType, AnswerItem

logger = logging.getLogger("SurveyNoiseSimulator")


class SurveyAnomalyType(str, Enum):
    STRAIGHT_LINER = "straight_liner"      # User who clicks same rating (all 3s or all 5s) across all scales without reading
    LOW_EFFORT_TEXT = "low_effort_text"    # User typing "N/A", "no comments", or "good" instead of long descriptions
    MINOR_TYPOS = "minor_typos"            # Realistic keyboard typing slip-ups or informal formatting


class HumanNoiseInjector:
    """
    Simulates real-world respondent survey fatigue and behavioral anomalies.
    """
    def __init__(
        self,
        straight_liner_rate: float = 0.04,
        low_effort_text_rate: float = 0.08,
        typo_rate: float = 0.04,
        seed: Optional[int] = None
    ):
        self.straight_liner_rate = max(0.0, min(1.0, straight_liner_rate))
        self.low_effort_text_rate = max(0.0, min(1.0, low_effort_text_rate))
        self.typo_rate = max(0.0, min(1.0, typo_rate))
        
        if seed is not None:
            random.seed(seed)
            
        # Realistic low-effort survey responses seen in genuine empirical datasets
        self.low_effort_phrases = [
            "N/A", "none", "NA", "no comments", "all good", "nothing specific",
            "good", "ok", "works fine", "no issues", "none at this time",
            "satisfied", "nil", "everything is fine", "na", "no suggestions"
        ]

    def _inject_typo(self, text: str) -> str:
        """Applies minor realistic keyboard typos (character swap or omission) to a sentence."""
        if len(text) < 5 or "http" in text or "@" in text:
            return text
            
        chars = list(text)
        idx = random.randint(1, len(chars) - 2)
        typo_kind = random.choice(["swap", "drop", "lower"])
        
        if typo_kind == "swap" and idx < len(chars) - 1:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif typo_kind == "drop":
            chars.pop(idx)
        elif typo_kind == "lower":
            return text.lower().rstrip(".")
            
        return "".join(chars)

    def inject_anomalies(
        self,
        dataset: List[FormAnswerSet],
        schema: FormSchema
    ) -> Tuple[List[FormAnswerSet], Dict[str, int]]:
        """
        Applies controlled human survey imperfections across a dataset batch.
        Returns the modified dataset alongside an audit count of applied anomalies.
        """
        total_n = len(dataset)
        if total_n == 0 or not schema:
            return dataset, {}

        stats: Dict[str, int] = {
            SurveyAnomalyType.STRAIGHT_LINER.value: 0,
            SurveyAnomalyType.LOW_EFFORT_TEXT.value: 0,
            SurveyAnomalyType.MINOR_TYPOS.value: 0
        }
        
        # 1. Straight-Liners (Speed-runners)
        scale_questions = [q for q in schema.all_questions if q.question_type == QuestionType.LINEAR_SCALE]
        if len(scale_questions) >= 2:
            num_straight_liners = int(round(total_n * self.straight_liner_rate))
            if num_straight_liners > 0 and total_n >= 5:
                # Select random respondent indices to act as speed runners
                straight_liner_indices = random.sample(range(total_n), min(total_n, max(1, num_straight_liners)))
                for idx in straight_liner_indices:
                    ans_set = dataset[idx]
                    # Speed-runner usually picks neutral middle rating or extreme 5s across every single scale
                    first_q = scale_questions[0]
                    min_v = first_q.scale_min or 1
                    max_v = first_q.scale_max or 5
                    chosen_uniform_val = random.choice([min_v + (max_v - min_v) // 2, max_v])
                    
                    for sq in scale_questions:
                        item = ans_set.get_answer(sq.id)
                        if item:
                            item.value = chosen_uniform_val
                        else:
                            ans_set.answers.append(AnswerItem(
                                question_id=sq.id,
                                question_title=sq.title,
                                value=chosen_uniform_val
                            ))
                    stats[SurveyAnomalyType.STRAIGHT_LINER.value] += 1
                    
        # 2. Low-Effort Text Replies & Minor Typos
        text_questions = [
            q for q in schema.all_questions 
            if q.question_type in (QuestionType.SHORT_ANSWER, QuestionType.PARAGRAPH)
            and not any(k in q.title.lower() for k in ["name", "email", "age", "phone", "contact"])
        ]
        
        for ans_set in dataset:
            for tq in text_questions:
                item = ans_set.get_answer(tq.id)
                if not item or not isinstance(item.value, str):
                    continue
                    
                # Low-effort text substitution
                if random.random() < self.low_effort_text_rate:
                    item.value = random.choice(self.low_effort_phrases)
                    stats[SurveyAnomalyType.LOW_EFFORT_TEXT.value] += 1
                # Minor typo simulation on normal long text
                elif random.random() < self.typo_rate:
                    item.value = self._inject_typo(item.value)
                    stats[SurveyAnomalyType.MINOR_TYPOS.value] += 1

        logger.info(f"🎭 Human Survey Noise simulation applied: {stats['straight_liner']} speed-runners, {stats['low_effort_text']} low-effort texts, {stats['minor_typos']} organic typos.")
        return dataset, stats

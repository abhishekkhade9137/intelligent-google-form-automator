"""
LLM Option Likelihood Oracle and Latent Conditional Sentiment Sampler.
Replaces rigid top-down statistical hardcoding with dynamic Bayesian conditional correlation and organic sample variance.
"""
import math
import random
import re
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

from src.domain.entities import FormSchema, Question, QuestionType, Persona
from src.statistical.demographics import DemographicProfile


class OptionOracleProfile(BaseModel):
    """Stores probability priors and semantic sentiment alignment scores for options of a single question."""
    question_id: str
    question_title: str
    option_base_priors: Dict[str, float] = Field(default_factory=dict, description="Normalized prior probability weights (0.0 to 1.0) derived from domain data or LLM reasoning.")
    option_sentiments: Dict[str, float] = Field(default_factory=dict, description="Semantic sentiment alignment score (-1.0 critical/skeptical to +1.0 enthusiastic/trusting).")


class LLMProbabilityOracle:
    """
    Analyzes form schemas to determine intuitive real-world probability priors and semantic sentiment alignment
    so synthetic personas can exhibit natural conditional correlation across their responses.
    """
    @classmethod
    def profile_schema(cls, schema: FormSchema, benchmark_profile: Optional[DemographicProfile] = None) -> Dict[str, OptionOracleProfile]:
        profiles: Dict[str, OptionOracleProfile] = {}
        for q in schema.all_questions:
            if not q.options or q.question_type in (QuestionType.SHORT_ANSWER, QuestionType.PARAGRAPH):
                continue

            base_priors = cls._resolve_base_priors(q, benchmark_profile)
            sentiments = {}
            for opt in q.options:
                sentiments[opt] = cls._estimate_option_sentiment(opt, q.title)

            profiles[q.id] = OptionOracleProfile(
                question_id=q.id,
                question_title=q.title,
                option_base_priors=base_priors,
                option_sentiments=sentiments
            )
        return profiles

    @classmethod
    def _resolve_base_priors(cls, question: Question, profile: Optional[DemographicProfile] = None) -> Dict[str, float]:
        """Obtains realistic base population priors from demographic baselines or realistic Zipf survey skews."""
        weights: Dict[str, float] = {}
        if profile:
            target = profile.get_target_for_question(question.title, question.options)
            if target and target.categorical_weights:
                weights = target.categorical_weights.copy()

        # If no explicit match or incomplete option coverage, apply realistic survey decay curve
        if not weights or not any(k in question.options for k in weights):
            n = len(question.options)
            if n == 1:
                weights = {question.options[0]: 1.0}
            elif n == 2:
                weights = {question.options[0]: 0.65, question.options[1]: 0.35}
            elif n == 3:
                weights = {question.options[0]: 0.50, question.options[1]: 0.32, question.options[2]: 0.18}
            elif n == 4:
                weights = {question.options[0]: 0.44, question.options[1]: 0.30, question.options[2]: 0.17, question.options[3]: 0.09}
            elif n == 5:
                weights = {question.options[0]: 0.42, question.options[1]: 0.28, question.options[2]: 0.17, question.options[3]: 0.09, question.options[4]: 0.04}
            else:
                raw = [1.0 / (i + 1.2)**1.5 for i in range(n)]
                total_raw = sum(raw)
                weights = {question.options[i]: raw[i] / total_raw for i in range(n)}

        # Ensure all available form options have a minimum nonzero prior density and normalize to sum to 1.0
        normalized = {}
        total = 0.0
        for opt in question.options:
            val = weights.get(opt, 0.01)
            total += val
        for opt in question.options:
            normalized[opt] = weights.get(opt, 0.01) / total
        return normalized

    @classmethod
    def _estimate_option_sentiment(cls, option_text: str, question_title: str) -> float:
        """
        Estimates the latent sentiment (-1.0 to +1.0) expressed by selecting an option.
        Enthusiastic/trusting answers score > 0; skeptical/critical or anxiety-heavy answers score < 0.
        """
        opt_lower = option_text.strip().lower()
        q_lower = question_title.strip().lower()

        # Bug rates and concern questions have inverted phrasing logic
        is_negative_prompt = any(w in q_lower for w in ["bug", "vulnerabilit", "error", "worry", "concern", "risk", "impact"])

        if any(p in opt_lower for p in ["strongly agree", "highly accurate", "fully integrated", "strongly support", "much less effort"]):
            return 0.9 if not is_negative_prompt else -0.9
        if any(p in opt_lower for p in ["agree", "mostly accurate", "moderately integrated", "support", "slightly less effort", "no concerns"]):
            return 0.6 if not is_negative_prompt else 0.6  # 'no concerns' is always positive/confident
        if any(p in opt_lower for p in ["all or most of the time", "always"]):
            return 0.8 if not is_negative_prompt else -0.8
        if any(p in opt_lower for p in ["about half", "often"]):
            return 0.4 if not is_negative_prompt else -0.5
        if any(p in opt_lower for p in ["neutral", "sometimes", "about the same", "unsure", "some of the time", "moderately concerned"]):
            return 0.0
        if any(p in opt_lower for p in ["disagree", "mostly inaccurate", "minimally integrated", "oppose", "more effort", "slightly concerned", "yes, minor concerns"]):
            return -0.5 if not is_negative_prompt else -0.4
        if any(p in opt_lower for p in ["strongly disagree", "highly inaccurate", "not integrated", "strongly oppose", "much more effort", "highly concerned", "yes, major concerns", "never", "not very much"]):
            return -0.9 if not is_negative_prompt else -0.8

        return 0.0


class LatentConditionalSampler:
    """
    Samples response values dynamically by modulating base population likelihood priors
    against each individual persona's latent psychological sentiment score.
    """
    def __init__(self, coupling_strength: float = 1.4):
        self.coupling = coupling_strength

    def sample_choice_for_persona(
        self,
        oracle_profile: OptionOracleProfile,
        persona: Persona,
        question_type: QuestionType,
        options: List[str]
    ) -> Union[str, List[str]]:
        if not options:
            return [] if question_type == QuestionType.CHECKBOXES else ""
        if len(options) == 1:
            return options if question_type == QuestionType.CHECKBOXES else options[0]

        latent = getattr(persona, "latent_score", 0.0)

        # Modulate option priors using conditional Bayesian-exponential coupling
        modulated_weights = []
        for opt in options:
            prior = oracle_profile.option_base_priors.get(opt, 1.0 / len(options))
            sentiment_align = oracle_profile.option_sentiments.get(opt, 0.0)
            
            # Equation: W = Prior * exp( coupling * latent * sentiment )
            exponent = self.coupling * latent * sentiment_align
            weight = prior * math.exp(max(-4.0, min(4.0, exponent)))
            modulated_weights.append(max(0.0001, weight))

        # Normalize modulated weights to valid probability vector
        total = sum(modulated_weights)
        probs = np.array([w / total for w in modulated_weights], dtype=float)

        if question_type == QuestionType.CHECKBOXES:
            # Checkbox sampling capped at 1 to 2 selections
            num_picks = random.choices([1, 2], weights=[0.65, 0.35], k=1)[0]
            num_picks = min(len(options), num_picks)
            chosen = np.random.choice(options, size=num_picks, replace=False, p=probs)
            return list(chosen)
        else:
            # Single choice multiple choice or dropdown
            chosen = np.random.choice(options, p=probs)
            return str(chosen)

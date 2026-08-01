"""
AI Data Generator Engine featuring Batch Synthesis & Resource Optimization.
Decouples statistical probability formulas from AI qualitative narrative text generation.
Integrates real-world demographic benchmarks, deterministic Hare-Niemann rebalancing,
multi-strategy authentic email synthesis, and survey anomaly simulation.
"""
import os
import json
import time
import random
import logging
import re
from typing import List, Optional, Dict, Any, Union, Callable
from faker import Faker

from src.domain.entities import FormSchema, Question, QuestionType, Persona, AnswerItem, FormAnswerSet
from src.synthesis.distributions import DistributionSampler, DistributionType
from src.synthesis.email_generator import MultiStrategyEmailGenerator
from src.synthesis.latent_oracle import LLMProbabilityOracle, LatentConditionalSampler, OptionOracleProfile
from src.statistical.demographics import DemographicProfile, get_profile_by_name
from src.statistical.rebalancer import HareNiemannRebalancer, ValidationReport
from src.statistical.noise_injector import HumanNoiseInjector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AIGenerationEngine")


class AIGenerationEngine:
    """
    Orchestrates high-efficiency batch answer completion with dynamic LLM probability likelihoods
    and Bayesian latent conditional persona correlation (no rigid hardcoded quotas).
    """
    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "glm-5.2:cloud",
        api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        faker_locale: str = "en_IN",
        temperature: float = 0.85,
        max_retries: int = 1,
        status_callback: Optional[Callable[[str], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.faker = Faker(faker_locale)
        self.sampler = DistributionSampler()
        self.status_callback = status_callback or (lambda msg: logger.info(msg))
        self.error_callback = error_callback or (lambda msg: logger.error(msg))
        self.demographic_context = ""
        
        # Instantiate statistical verification, oracle likelihoods, and diversity engines
        self.email_generator = MultiStrategyEmailGenerator()
        self.rebalancer = HareNiemannRebalancer()
        self.noise_injector = HumanNoiseInjector()
        self.latent_sampler = LatentConditionalSampler(coupling_strength=1.4)

    def generate_batch_campaign(
        self,
        schema: FormSchema,
        count: int,
        context: str = "",
        demographic_guidance: Optional[str] = None,
        benchmark_profile: Optional[Union[str, DemographicProfile]] = None,
        apply_noise: bool = True,
        apply_rebalancing: bool = False  # Disabled by default to preserve authentic stochastic sample variance
    ) -> List[FormAnswerSet]:
        """
        Synthesizes an entire organic campaign dataset in memory using conditional latent persona sentiment
        and LLM probability option profiling, avoiding rigid artificial percentage enforcement.
        """
        self.status_callback(f"📦 Pre-computing batch of {count} developer profiles using Latent Sentiment & Conditional Oracle Sampler...")
        start_time = time.time()
        
        # Resolve real-world demographic benchmark profile if provided as string ID/name
        prof_obj: Optional[DemographicProfile] = None
        if isinstance(benchmark_profile, str):
            prof_obj = get_profile_by_name(benchmark_profile)
            if prof_obj:
                self.status_callback(f"📊 Aligned LLM Likelihood Oracle to demographic baseline: '{prof_obj.name}'")
        elif isinstance(benchmark_profile, DemographicProfile):
            prof_obj = benchmark_profile
            self.status_callback(f"📊 Aligned LLM Likelihood Oracle to custom profile: '{prof_obj.name}'")

        batch_results: List[FormAnswerSet] = []
        used_names: List[str] = []
        self.email_generator.reset_history()
        
        # Profile schema options with LLM probability likelihoods and sentiment alignment vectors
        self.status_callback("🧠 Profiling form options with LLM Option Likelihood Oracle for authentic conditional variance...")
        oracle_profiles = LLMProbabilityOracle.profile_schema(schema, benchmark_profile=prof_obj)

        for i in range(count):
            persona = self.create_guided_persona(context, demographic_guidance, used_names, skip_llm_backstory=True)
            used_names.append(persona.name)
            
            answer_set = self.generate_form_answers(schema, persona, context, distribution_targets=oracle_profiles)
            batch_results.append(answer_set)

        # Optional deterministic rebalancing layer if strict quota overriding is explicitly requested
        if apply_rebalancing and batch_results:
            self.status_callback("⚖️ Applying Hare-Niemann quota rebalancing (strict enforcement)...")
            batch_results, report = self.rebalancer.rebalance_dataset(batch_results, schema, profile=prof_obj)
            if report.total_rebalanced_entries > 0:
                self.status_callback(f"✅ Rebalanced {report.total_rebalanced_entries} entries (Mean TVD: {report.mean_final_tvd:.4f}).")

        # Human Survey Noise Simulation Layer
        if apply_noise and batch_results:
            self.status_callback("🎭 Applying human survey anomalies (speed-running, low-effort replies, minor typos)...")
            batch_results, stats = self.noise_injector.inject_anomalies(batch_results, schema)
            self.status_callback(f"✅ Injected noise: {stats.get('straight_liner', 0)} speed-runners, {stats.get('low_effort_text', 0)} short texts, {stats.get('minor_typos', 0)} typos.")
            
        elapsed = round(time.time() - start_time, 2)
        self.status_callback(f"🎉 Pre-flight conditional dataset synthesis completed in {elapsed}s! Ready for browser execution.")
        return batch_results

    def generate_form_answers(
        self,
        schema: FormSchema,
        persona: Persona,
        context: str = "",
        distribution_targets: Optional[Dict[str, Any]] = None
    ) -> FormAnswerSet:
        """
        Generates a complete set of answers for a single persona against a form schema
        utilizing conditional latent sentiment sampling for coherent cross-tabulation.
        """
        distribution_targets = distribution_targets or {}
        answers: List[AnswerItem] = []

        for q in schema.all_questions:
            val = None

            # 1. Open text descriptive or demographic fields (Short Answer / Paragraph)
            if q.question_type in (QuestionType.SHORT_ANSWER, QuestionType.PARAGRAPH):
                q_lower = q.title.lower()
                if re.search(r'\bname\b', q_lower):
                    val = persona.name
                elif re.search(r'\b(email|e-mail|mail)\b', q_lower):
                    val = self._generate_email_for_persona(persona)
                elif re.search(r'\bage\b', q_lower) or "year of birth" in q_lower:
                    val = str(persona.age)
                elif re.search(r'\b(role|profession|occupation|designation)\b', q_lower):
                    val = persona.occupation or "Student / Academic"
                elif re.search(r'\b(city|location|campus|where)\b', q_lower):
                    val = "VIT Vellore"
                elif q.id in distribution_targets and not isinstance(distribution_targets[q.id], OptionOracleProfile) and distribution_targets[q.id] is not None:
                    val = distribution_targets[q.id]
                else:
                    phrases = [
                        "Essential for speeding up Python boilerplate and debugging syntax errors in coursework.",
                        "Works remarkably well for C++ algorithm practice and refactoring functions during lab practicals.",
                        "Saves countless hours during full-stack web development projects at university.",
                        "Highly accurate for standard libraries, though complex design logic still needs careful testing.",
                        "Indispensable tool for daily coding projects and faster bug resolution."
                    ]
                    val = random.choice(phrases)
            else:
                # 2. Structured Choice & Rating Fields (Multiple Choice, Checkboxes, Dropdown, Linear Scale)
                target_or_oracle = distribution_targets.get(q.id)
                if isinstance(target_or_oracle, OptionOracleProfile):
                    val = self.latent_sampler.sample_choice_for_persona(
                        oracle_profile=target_or_oracle,
                        persona=persona,
                        question_type=q.question_type,
                        options=q.options
                    )
                elif target_or_oracle is not None and not isinstance(target_or_oracle, OptionOracleProfile):
                    val = target_or_oracle
                else:
                    if q.question_type == QuestionType.LINEAR_SCALE:
                        val = self.sampler.sample_linear_scale(q.scale_min or 1, q.scale_max or 5, DistributionType.NORMAL)
                    elif q.question_type == QuestionType.CHECKBOXES and q.options:
                        val = self.sampler.sample_checkboxes(q.options, avg_selection_count=1, max_selection_count=2)
                    elif q.options:
                        val = self._sample_fallback_value(q, None, persona)

            answers.append(AnswerItem(question_id=q.id, question_title=q.title, value=val))

        return FormAnswerSet(persona=persona, answers=answers)

    def create_guided_persona(
        self,
        context: str = "",
        demographic_guidance: Optional[str] = None,
        used_names: Optional[List[str]] = None,
        skip_llm_backstory: bool = True
    ) -> Persona:
        used_names = used_names or []
        self.demographic_context = str(demographic_guidance or "").lower()
        name = "Rahul Verma"
        for _ in range(30):
            candidate = self.faker.name()
            if candidate not in used_names and not any(title in candidate for title in ["Dr.", "Prof.", "Mr.", "Mrs.", "Ms."]):
                name = candidate
                break

        age = random.randint(19, 22)
        occupations = [
            "B.Tech Computer Science Undergrad (VIT Vellore)",
            "B.Tech AI & Data Science Student (VIT)",
            "Information Technology Student (VIT Vellore)",
            "Computer Science Engineering Undergrad",
            "Software Engineering Student"
        ]
        selected_occupation = random.choice(occupations)

        # Assign continuous psychological latent sentiment score from Gaussian population curve
        latent_score = round(max(-1.0, min(1.0, random.gauss(0.15, 0.45))), 2)
        if latent_score >= 0.35:
            selected_sentiment = "Enthusiastic & Pragmatic (Power user, trusting of AI precision)"
            traits = ["tech-savvy", "early adopter", "efficiency-driven"]
        elif latent_score >= -0.20:
            selected_sentiment = "Balanced & Moderate (Appreciates productivity gains but careful with code verification)"
            traits = ["pragmatic coder", "code reviewer", "balanced user"]
        else:
            selected_sentiment = "Cautious & Tech Skeptic (Concerned over dependency, bugs, and data privacy leaks)"
            traits = ["security-conscious", "rigorous verification", "traditional problem-solver"]

        background_text = f"A computer science undergraduate at VIT Vellore with a {selected_sentiment.lower()} methodology."

        return Persona(
            name=name,
            age=age,
            occupation=selected_occupation,
            background=background_text.strip(),
            sentiment=selected_sentiment,
            traits=traits,
            latent_score=latent_score
        )

    def _generate_email_for_persona(self, persona: Persona) -> str:
        """Generates authentic multi-pattern email addresses using MultiStrategyEmailGenerator."""
        return self.email_generator.generate_email(persona.name)

    def _sample_fallback_value(self, question: Question, assigned_target: Any, persona: Optional[Persona] = None) -> Any:
        if question.options and question.question_type in (QuestionType.MULTIPLE_CHOICE, QuestionType.DROPDOWN):
            return random.choice(question.options)
        return ""


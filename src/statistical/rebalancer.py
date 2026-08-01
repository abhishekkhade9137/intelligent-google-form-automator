"""
Deterministic Dataset Validator & Rebalancer Engine.
Decouples statistical accuracy from AI text generation by employing rigorous mathematical formulas
(Total Variation Distance, Chi-Square, and Hare-Niemann Largest Remainder Method) to verify and
adjust in-memory datasets before browser submission occurs.
"""
import math
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from src.domain.entities import FormAnswerSet, FormSchema, Question, QuestionType, AnswerItem
from src.statistical.demographics import DemographicProfile, FieldDistributionTarget

logger = logging.getLogger("DatasetValidator")


class FieldValidationResult(BaseModel):
    """Audit summary of statistical verification and rebalancing for a single field."""
    question_id: str
    question_title: str
    initial_tvd: float = Field(description="Total Variation Distance before rebalancing (0.0 = perfect match).")
    final_tvd: float = Field(description="Total Variation Distance after deterministic rebalancing.")
    rebalanced_count: int = Field(description="Number of respondent entries modified to achieve empirical alignment.")
    target_distribution: Dict[str, float] = Field(default_factory=dict)
    achieved_distribution: Dict[str, float] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """Comprehensive dataset verification report across an entire survey campaign."""
    total_submissions: int
    fields_evaluated: int
    total_rebalanced_entries: int
    passed_initial_tolerance: bool
    field_results: List[FieldValidationResult] = Field(default_factory=list)

    @property
    def mean_final_tvd(self) -> float:
        if not self.field_results:
            return 0.0
        return sum(f.final_tvd for f in self.field_results) / len(self.field_results)


class StatisticalDivergenceEvaluator:
    """
    Computes standard empirical probability divergence metrics between actual and target distributions.
    """
    @staticmethod
    def normalize_weights(weights_dict: Dict[str, float]) -> Dict[str, float]:
        """Normalizes a weight dictionary so probabilities sum exactly to 1.0."""
        total = sum(weights_dict.values())
        if total <= 0:
            count = max(1, len(weights_dict))
            return {k: 1.0 / count for k in weights_dict}
        return {k: v / total for k, v in weights_dict.items()}

    @classmethod
    def compute_tvd(cls, actual_dist: Dict[str, float], target_dist: Dict[str, float]) -> float:
        """
        Computes Total Variation Distance (TVD) = 0.5 * SUM(|P(k) - Q(k)|).
        A TVD of 0.0 means identical distribution; 1.0 means completely disjoint.
        """
        norm_actual = cls.normalize_weights(actual_dist)
        norm_target = cls.normalize_weights(target_dist)
        
        all_keys = set(norm_actual.keys()).union(set(norm_target.keys()))
        tvd_sum = sum(abs(norm_actual.get(k, 0.0) - norm_target.get(k, 0.0)) for k in all_keys)
        return round(0.5 * tvd_sum, 4)

    @classmethod
    def compute_chi_square_distance(cls, actual_counts: Dict[str, int], target_weights: Dict[str, float], total_samples: int) -> float:
        """
        Computes Pearson's Chi-Square test statistic for goodness of fit.
        """
        if total_samples <= 0 or not target_weights:
            return 0.0
            
        norm_target = cls.normalize_weights(target_weights)
        chi_sq = 0.0
        for k, p_target in norm_target.items():
            expected = max(0.001, p_target * total_samples)
            observed = float(actual_counts.get(k, 0))
            chi_sq += ((observed - expected) ** 2) / expected
        return round(chi_sq, 4)


class HareNiemannRebalancer:
    """
    Applies the Hare-Niemann Quota Allocation algorithm (Largest Remainder Method) to deterministically
    reallocate answers across a batch so empirical demographic distributions are met with minimal rounding deviation.
    """
    def __init__(self, tolerance_tvd: float = 0.02, seed: Optional[int] = None):
        self.tolerance_tvd = tolerance_tvd
        if seed is not None:
            random.seed(seed)

    def calculate_exact_quota(self, target_weights: Dict[str, float], total_count: int) -> Dict[str, int]:
        """
        Computes mathematically optimal integer frequency counts for each category such that
        the sum equals total_count and Total Variation Distance from target_weights is minimized.
        """
        if not target_weights or total_count <= 0:
            return {}
            
        norm_weights = StatisticalDivergenceEvaluator.normalize_weights(target_weights)
        quotas: Dict[str, float] = {k: p * total_count for k, p in norm_weights.items()}
        
        # 1. Base floor allocations
        allocations: Dict[str, int] = {k: int(math.floor(q)) for k, q in quotas.items()}
        allocated_sum = sum(allocations.values())
        remainder_count = total_count - allocated_sum
        
        if remainder_count > 0:
            # 2. Sort by fractional remainders descending
            remainders = [(k, quotas[k] - allocations[k]) for k in quotas]
            remainders.sort(key=lambda x: x[1], reverse=True)
            
            # 3. Distribute unallocated remainder slots to largest fractional components
            for i in range(remainder_count):
                key = remainders[i % len(remainders)][0]
                allocations[key] += 1
                
        return allocations

    def _discretize_normal_scale(self, min_val: int, max_val: int, mean: float, std: float) -> Dict[str, float]:
        """Converts continuous normal mean/std target into discrete categorical probability weights for rating scales."""
        weights = {}
        for x in range(min_val, max_val + 1):
            # Normal probability density approximation for discrete integer bucket
            exponent = -((x - mean) ** 2) / (2 * max(0.01, std ** 2))
            prob = (1.0 / (max(0.01, std) * math.sqrt(2 * math.pi))) * math.exp(exponent)
            weights[str(x)] = max(0.001, prob)
        return weights

    def _generate_default_skewed_weights(self, options: List[str]) -> Dict[str, float]:
        """Generates a realistic skewed survey distribution curve instead of an artificial flat uniform split."""
        n = len(options)
        if n == 1:
            return {options[0]: 100.0}
        elif n == 2:
            return {options[0]: 65.0, options[1]: 35.0}
        elif n == 3:
            return {options[0]: 50.0, options[1]: 32.0, options[2]: 18.0}
        elif n == 4:
            return {options[0]: 44.0, options[1]: 30.0, options[2]: 17.0, options[3]: 9.0}
        elif n == 5:
            return {options[0]: 42.0, options[1]: 28.0, options[2]: 17.0, options[3]: 9.0, options[4]: 4.0}
        else:
            raw = [1.0 / (i + 1.2)**1.5 for i in range(n)]
            total = sum(raw)
            return {options[i]: round((raw[i] / total) * 100.0, 2) for i in range(n)}

    def rebalance_dataset(
        self,
        dataset: List[FormAnswerSet],
        schema: FormSchema,
        profile: Optional[DemographicProfile] = None
    ) -> Tuple[List[FormAnswerSet], ValidationReport]:
        """
        Verifies and rebalances an entire in-memory dataset against targeted demographic benchmarks
        prior to automated Google Form browser interaction.
        """
        total_n = len(dataset)
        if total_n == 0 or not schema:
            return dataset, ValidationReport(total_submissions=0, fields_evaluated=0, total_rebalanced_entries=0, passed_initial_tolerance=True)

        field_results: List[FieldValidationResult] = []
        total_rebalanced_count = 0
        all_passed_initially = True

        for question in schema.all_questions:
            # Checkboxes allow multiple selections per respondent; skip single-choice categorical quota rebalancing
            if question.question_type == QuestionType.CHECKBOXES:
                continue

            # Check if this field has target categorical or scale benchmarks
            target_weights: Dict[str, float] = {}
            if profile:
                t_prof = profile.get_target_for_question(question.title, question.options)
                if t_prof:
                    if t_prof.categorical_weights:
                        target_weights = t_prof.categorical_weights.copy()
                    elif t_prof.numerical_mean is not None and question.scale_min is not None and question.scale_max is not None:
                        std_val = t_prof.numerical_std or max(0.5, (question.scale_max - question.scale_min) / 4.0)
                        target_weights = self._discretize_normal_scale(question.scale_min, question.scale_max, t_prof.numerical_mean, std_val)

            # Fallback default target if options exist and no external profile match found
            if not target_weights:
                if question.options and question.question_type in (QuestionType.MULTIPLE_CHOICE, QuestionType.DROPDOWN):
                    target_weights = self._generate_default_skewed_weights(question.options)
                elif question.question_type == QuestionType.LINEAR_SCALE and question.scale_min is not None and question.scale_max is not None:
                    mean = (question.scale_min + question.scale_max) / 2.0
                    std = max(1.0, (question.scale_max - question.scale_min) / 3.0)
                    target_weights = self._discretize_normal_scale(question.scale_min, question.scale_max, mean, std)

            if not target_weights:
                continue  # Skip free text or un-weighted open-ended fields

            # Normalize option keys to strings for comparison
            norm_target_weights = {str(k): float(v) for k, v in target_weights.items()}
            
            # Map existing answers in dataset to current observed frequency table
            observed_counts: Dict[str, int] = {k: 0 for k in norm_target_weights}
            answer_mapping: Dict[str, List[int]] = {k: [] for k in norm_target_weights}
            unmapped_indices: List[int] = []

            for idx, ans_set in enumerate(dataset):
                ans_item = ans_set.get_answer(question.id)
                val_str = str(ans_item.value) if (ans_item and ans_item.value is not None) else ""
                
                # Match value against known target keys (exact or substring matching)
                matched_key = None
                if val_str in norm_target_weights:
                    matched_key = val_str
                else:
                    # Fuzzy / substring matching for options
                    for t_key in norm_target_weights:
                        if t_key.lower() in val_str.lower() or val_str.lower() in t_key.lower():
                            if t_key != "0" or val_str == "0":
                                matched_key = t_key
                                break
                                
                if matched_key:
                    observed_counts[matched_key] = observed_counts.get(matched_key, 0) + 1
                    answer_mapping[matched_key].append(idx)
                else:
                    unmapped_indices.append(idx)

            # Calculate initial TVD
            observed_freq = {k: (c / total_n) for k, c in observed_counts.items()}
            initial_tvd = StatisticalDivergenceEvaluator.compute_tvd(observed_freq, norm_target_weights)

            if initial_tvd > self.tolerance_tvd or len(unmapped_indices) > 0:
                all_passed_initially = False
                
                # Compute exact Hare-Niemann target quota counts
                target_quotas = self.calculate_exact_quota(norm_target_weights, total_n)
                
                # Identify over-represented pools and under-represented deficit needs
                surplus_indices: List[int] = list(unmapped_indices) # Unmapped go first into surplus reassignment
                for k, indices in answer_mapping.items():
                    target_c = target_quotas.get(k, 0)
                    if len(indices) > target_c:
                        excess_count = len(indices) - target_c
                        # Randomly pick excess entries to pull out of overrepresented bucket
                        to_remove = random.sample(indices, excess_count)
                        surplus_indices.extend(to_remove)
                        for r_idx in to_remove:
                            indices.remove(r_idx)

                # Assign surplus entries to under-represented deficit buckets
                field_rebalanced_count = 0
                random.shuffle(surplus_indices)
                
                for k, target_c in target_quotas.items():
                    current_c = len(answer_mapping.get(k, []))
                    while current_c < target_c and surplus_indices:
                        idx_to_fill = surplus_indices.pop()
                        
                        # Apply new deterministic target value to respondent's answer item
                        new_val: Any = k
                        if question.question_type == QuestionType.LINEAR_SCALE:
                            try:
                                new_val = int(k)
                            except ValueError:
                                new_val = question.scale_min or 3
                                
                        existing_item = dataset[idx_to_fill].get_answer(question.id)
                        if existing_item:
                            existing_item.value = new_val
                        else:
                            dataset[idx_to_fill].answers.append(AnswerItem(
                                question_id=question.id,
                                question_title=question.title,
                                value=new_val
                            ))
                            
                        answer_mapping[k].append(idx_to_fill)
                        current_c += 1
                        field_rebalanced_count += 1
                        total_rebalanced_count += 1

                # Calculate post-rebalance TVD
                final_counts = {k: len(indices) for k, indices in answer_mapping.items()}
                final_freq = {k: (c / total_n) for k, c in final_counts.items()}
                final_tvd = StatisticalDivergenceEvaluator.compute_tvd(final_freq, norm_target_weights)
                
                logger.info(f"⚖️ Rebalanced field '{question.title}': TVD reduced from {initial_tvd:.4f} ➡️ {final_tvd:.4f} ({field_rebalanced_count} items reassigned).")
            else:
                final_tvd = initial_tvd
                final_freq = observed_freq

            field_results.append(FieldValidationResult(
                question_id=question.id,
                question_title=question.title,
                initial_tvd=initial_tvd,
                final_tvd=final_tvd,
                rebalanced_count=int(round(initial_tvd * total_n)) if initial_tvd > self.tolerance_tvd else 0,
                target_distribution=StatisticalDivergenceEvaluator.normalize_weights(norm_target_weights),
                achieved_distribution=final_freq
            ))

        report = ValidationReport(
            total_submissions=total_n,
            fields_evaluated=len(field_results),
            total_rebalanced_entries=total_rebalanced_count,
            passed_initial_tolerance=all_passed_initially,
            field_results=field_results
        )
        
        return dataset, report

"""
Statistical distributions engine for generating realistic response datasets across form campaigns.
Prevents clustering and naive random noise by applying formal mathematical probability distributions.
"""
import math
import random
from enum import Enum
from typing import List, Dict, Any, Optional, Union
import numpy as np


class DistributionType(str, Enum):
    NORMAL = "normal"          # Bell curve around a target mean (e.g., standard customer sentiment)
    UNIFORM = "uniform"        # Equal probability across all choices
    SKEWED_HIGH = "skewed_high"  # Beta distribution heavily weighted towards top scores (e.g., enthusiastic userbase)
    SKEWED_LOW = "skewed_low"    # Beta distribution heavily weighted towards lower scores (e.g., complaint survey)
    WEIGHTED = "weighted"        # Explicit user-defined percentage weights for options


class DistributionSampler:
    """
    Samples target values for quantitative and closed-ended form fields using rigorous statistical methods.
    These sampled targets serve as coherent constraints for the LLM when writing accompanying textual explanations.
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def sample_linear_scale(
        self,
        min_val: int,
        max_val: int,
        distribution: DistributionType = DistributionType.NORMAL,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None
    ) -> int:
        """
        Samples an integer rating within [min_val, max_val] using the requested probability density curve.
        """
        span = max_val - min_val
        if span <= 0:
            return min_val

        if distribution == DistributionType.NORMAL:
            mean = target_mean if target_mean is not None else (min_val + max_val) / 2.0
            std = target_std if target_std is not None else max(1.0, span / 4.0)
            # Sample from normal and clamp to valid range
            val = int(round(np.random.normal(loc=mean, scale=std)))
            return max(min_val, min(max_val, val))

        elif distribution == DistributionType.SKEWED_HIGH:
            # Beta(4, 1.5) skews towards upper bound
            beta_sample = np.random.beta(a=4.0, b=1.5)
            val = min_val + int(round(beta_sample * span))
            return max(min_val, min(max_val, val))

        elif distribution == DistributionType.SKEWED_LOW:
            # Beta(1.5, 4) skews towards lower bound
            beta_sample = np.random.beta(a=1.5, b=4.0)
            val = min_val + int(round(beta_sample * span))
            return max(min_val, min(max_val, val))

        else:
            # Default fallback to discrete uniform
            return random.randint(min_val, max_val)

    def sample_multiple_choice(
        self,
        options: List[str],
        weights: Optional[List[float]] = None,
        distribution: DistributionType = DistributionType.UNIFORM
    ) -> str:
        """
        Selects a categorical choice according to specified probability weights or distributions.
        """
        if not options:
            return ""
        if len(options) == 1:
            return options[0]

        if weights and len(weights) == len(options):
            # Normalize weights so they sum to 1.0
            norm_weights = np.array(weights, dtype=float)
            norm_weights = norm_weights / norm_weights.sum()
            return np.random.choice(options, p=norm_weights)

        if distribution == DistributionType.SKEWED_HIGH:
            # Give higher indices exponentially higher weight
            raw_weights = [math.exp(i / 2.0) for i in range(len(options))]
            norm_weights = np.array(raw_weights) / sum(raw_weights)
            return np.random.choice(options, p=norm_weights)

        elif distribution == DistributionType.SKEWED_LOW:
            # Give lower indices exponentially higher weight
            raw_weights = [math.exp((len(options) - 1 - i) / 2.0) for i in range(len(options))]
            norm_weights = np.array(raw_weights) / sum(raw_weights)
            return np.random.choice(options, p=norm_weights)

        # Default Uniform
        return random.choice(options)

    def sample_checkboxes(
        self,
        options: List[str],
        avg_selection_count: int = 1,
        max_selection_count: int = 2,
        weights: Optional[List[float]] = None
    ) -> List[str]:
        """
        Samples a realistic subset of checkbox options (capped at 1 or 2 to prevent selecting everything).
        """
        if not options:
            return []
        
        # Sample number of items to pick (65% select 1 option, 35% select 2 options)
        num_to_pick = random.choices([1, 2], weights=[0.65, 0.35], k=1)[0]
        num_to_pick = max(1, min(len(options), min(max_selection_count, num_to_pick)))

        if weights and len(weights) == len(options):
            norm_weights = np.array(weights, dtype=float)
            norm_weights = norm_weights / norm_weights.sum()
            selected = np.random.choice(options, size=num_to_pick, replace=False, p=norm_weights)
        else:
            selected = random.sample(options, k=num_to_pick)

        return list(selected)

    def generate_campaign_plan(
        self,
        question_id: str,
        question_type: str,
        total_submissions: int,
        options: Optional[List[str]] = None,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        distribution: DistributionType = DistributionType.NORMAL,
        weights: Optional[List[float]] = None
    ) -> List[Union[str, int, List[str]]]:
        """
        Pre-computes the exact mathematical distribution array for a single field across the entire batch
        of N submissions, guaranteeing perfect summary analytics and chart visualization.
        """
        results = []
        for _ in range(total_submissions):
            if question_type == "linear_scale" and min_val is not None and max_val is not None:
                results.append(self.sample_linear_scale(min_val, max_val, distribution))
            elif question_type in ("multiple_choice", "dropdown") and options:
                results.append(self.sample_multiple_choice(options, weights, distribution))
            elif question_type == "checkboxes" and options:
                results.append(self.sample_checkboxes(options, avg_selection_count=1, max_selection_count=2, weights=weights))
            else:
                results.append(None)  # Text fields will be filled dynamically by Gemini
        return results

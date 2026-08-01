"""
Multi-dimensional psychometric trait correlation and Gaussian Copula sampling engine.
Generates realistic respondent behavior profiles where technical enthusiasm, privacy risk aversion,
and workplace pragmatism correlate naturally without rigid rule-based logic.
"""
import math
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.domain.entities import PsychometricProfile

logger = logging.getLogger("PsychometricCopulaEngine")


class GaussianCopulaSampler:
    """
    Implements Gaussian Copula sampling via Cholesky decomposition of trait correlation matrices.
    Enforces authentic psychological consistency across independent survey questions.
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Standard 3x3 correlation matrix for:
        # [0]: Tech Enthusiasm, [1]: Risk & Privacy Skepticism, [2]: Pragmatic Workflow
        # Tech Enthusiasm correlates negatively with Risk Skepticism (-0.55)
        # Tech Enthusiasm correlates positively with Pragmatic Workflow (+0.40)
        # Risk Skepticism correlates slightly negatively with Pragmatic Workflow (-0.25)
        self.default_correlation_matrix = np.array([
            [ 1.00, -0.55,  0.40],
            [-0.55,  1.00, -0.25],
            [ 0.40, -0.25,  1.00]
        ], dtype=float)

        self._cholesky_lower: Optional[np.ndarray] = None
        try:
            self._cholesky_lower = np.linalg.cholesky(self.default_correlation_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Correlation matrix not positive definite, defaulting to identity.")
            self._cholesky_lower = np.eye(3)

    def sample_trait_vector(self) -> Tuple[float, float, float]:
        """
        Samples a 3D vector of correlated standard normal random variables (Z-scores).
        Returns: (tech_enthusiasm, risk_skepticism, pragmatic_workflow)
        """
        # Independent standard normal samples
        z_ind = np.random.normal(loc=0.0, scale=1.0, size=3)
        # Apply Cholesky transformation
        if self._cholesky_lower is not None:
            z_corr = self._cholesky_lower @ z_ind
        else:
            z_corr = z_ind

        # Bound extreme tails between -3.0 and +3.0
        tech_enthusiasm = float(np.clip(z_corr[0], -3.0, 3.0))
        risk_skepticism = float(np.clip(z_corr[1], -3.0, 3.0))
        pragmatic_workflow = float(np.clip(z_corr[2], -3.0, 3.0))

        return round(tech_enthusiasm, 3), round(risk_skepticism, 3), round(pragmatic_workflow, 3)

    def generate_profile(self, persona_id: str) -> PsychometricProfile:
        """Generates a complete psychometric persona profile with correlated behavior traits."""
        tech, risk, prag = self.sample_trait_vector()
        return PsychometricProfile(
            persona_id=persona_id,
            tech_enthusiasm=tech,
            risk_skepticism=risk,
            pragmatic_workflow=prag,
            trait_weights={
                "tech_enthusiasm": tech,
                "risk_skepticism": risk,
                "pragmatic_workflow": prag
            }
        )

    @classmethod
    def compute_conditional_weight(
        cls,
        base_prior: float,
        option_sentiment: float,
        trait_score: float,
        sensitivity: float = 0.75
    ) -> float:
        """
        Modulates an option's probability prior using exponential weighting against a latent trait score.
        If trait_score aligns with option_sentiment (both positive or both negative), probability exponentially increases.
        """
        if base_prior <= 0.0:
            return 0.0
        # Alignment product: e.g., (+2.0 tech enthusiasm) * (+0.9 option sentiment) = +1.8 -> boosted probability
        alignment_multiplier = math.exp(sensitivity * trait_score * option_sentiment)
        return max(1e-5, base_prior * alignment_multiplier)

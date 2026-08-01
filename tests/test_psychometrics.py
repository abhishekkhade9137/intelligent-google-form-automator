import pytest
import numpy as np
from src.statistical.psychometrics import GaussianCopulaSampler
from src.domain.entities import PsychometricProfile


def test_gaussian_copula_sampling_bounds():
    sampler = GaussianCopulaSampler(seed=42)
    tech, risk, prag = sampler.sample_trait_vector()
    
    assert -3.0 <= tech <= 3.0
    assert -3.0 <= risk <= 3.0
    assert -3.0 <= prag <= 3.0


def test_profile_generation():
    sampler = GaussianCopulaSampler(seed=101)
    profile = sampler.generate_profile("test-persona-xyz")
    
    assert isinstance(profile, PsychometricProfile)
    assert profile.persona_id == "test-persona-xyz"
    assert "tech_enthusiasm" in profile.trait_weights
    assert "risk_skepticism" in profile.trait_weights
    assert "pragmatic_workflow" in profile.trait_weights


def test_conditional_weight_modulation():
    base_prior = 0.25
    # Positive alignment (enthusiastic respondent viewing enthusiastic option)
    boosted = GaussianCopulaSampler.compute_conditional_weight(
        base_prior=base_prior,
        option_sentiment=1.0,
        trait_score=2.0,
        sensitivity=0.5
    )
    assert boosted > base_prior

    # Negative alignment (enthusiastic respondent viewing opposed option)
    reduced = GaussianCopulaSampler.compute_conditional_weight(
        base_prior=base_prior,
        option_sentiment=-1.0,
        trait_score=2.0,
        sensitivity=0.5
    )
    assert reduced < base_prior

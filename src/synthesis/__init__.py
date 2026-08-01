"""
Synthesis and sampling layer.
Orchestrates persona campaign generation, multi-strategy email synthesis, probability distribution sampling,
and conditional Bayesian sentiment correlation.
"""
from src.synthesis.distributions import DistributionType, DistributionSampler
from src.synthesis.email_generator import EmailStrategy, MultiStrategyEmailGenerator
from src.synthesis.latent_oracle import OptionOracleProfile, LLMProbabilityOracle, LatentConditionalSampler
from src.synthesis.campaign_synthesizer import AIGenerationEngine

__all__ = [
    "DistributionType",
    "DistributionSampler",
    "EmailStrategy",
    "MultiStrategyEmailGenerator",
    "OptionOracleProfile",
    "LLMProbabilityOracle",
    "LatentConditionalSampler",
    "AIGenerationEngine",
]

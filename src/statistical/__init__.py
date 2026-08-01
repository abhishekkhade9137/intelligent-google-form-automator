"""
Statistical, demographic benchmarking, and psychometric trait correlation layer.
Provides real-world demographic survey baselines, Hare-Niemann quota rebalancing, Total Variation Distance auditing,
controlled survey anomaly simulation, and Gaussian Copula multi-dimensional trait correlation.
"""
from src.statistical.demographics import (
    FieldDistributionTarget,
    DemographicProfile,
    PRESET_PROFILES,
    STACK_OVERFLOW_2024_DEV_SURVEY,
    INDIAN_UNIVERSITY_DEMOGRAPHICS,
    GLOBAL_CONSUMER_SENTIMENT,
    BenchmarkAPIFetcher,
    get_profile_by_name,
)
from src.statistical.rebalancer import (
    FieldValidationResult,
    ValidationReport,
    HareNiemannRebalancer,
)
from src.statistical.noise_injector import (
    SurveyAnomalyType,
    HumanNoiseInjector,
)
from src.statistical.psychometrics import (
    GaussianCopulaSampler,
)

__all__ = [
    "FieldDistributionTarget",
    "DemographicProfile",
    "PRESET_PROFILES",
    "STACK_OVERFLOW_2024_DEV_SURVEY",
    "INDIAN_UNIVERSITY_DEMOGRAPHICS",
    "GLOBAL_CONSUMER_SENTIMENT",
    "BenchmarkAPIFetcher",
    "get_profile_by_name",
    "FieldValidationResult",
    "ValidationReport",
    "HareNiemannRebalancer",
    "SurveyAnomalyType",
    "HumanNoiseInjector",
    "GaussianCopulaSampler",
]

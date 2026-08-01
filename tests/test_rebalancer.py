"""
Unit tests for deterministic mathematical dataset verification and Hare-Niemann rebalancing.
"""
import pytest
from src.statistical.rebalancer import StatisticalDivergenceEvaluator, HareNiemannRebalancer
from src.domain.entities import FormAnswerSet, Persona, AnswerItem, FormSchema, Question, QuestionType, FormPage


def test_tvd_computation():
    dist_a = {"A": 50.0, "B": 50.0}
    dist_b = {"A": 50.0, "B": 50.0}
    assert StatisticalDivergenceEvaluator.compute_tvd(dist_a, dist_b) == 0.0
    
    dist_c = {"A": 100.0, "B": 0.0}
    dist_d = {"A": 0.0, "B": 100.0}
    assert StatisticalDivergenceEvaluator.compute_tvd(dist_c, dist_d) == 1.0


def test_hare_niemann_exact_quota():
    rebalancer = HareNiemannRebalancer(seed=42)
    weights = {"Python": 70.0, "Java": 20.0, "Other": 10.0}
    
    # For N = 25 submissions
    quotas = rebalancer.calculate_exact_quota(weights, 25)
    assert sum(quotas.values()) == 25
    assert quotas["Python"] == 18  # 70% of 25 = 17.5 -> rounded up to 18
    assert quotas["Java"] == 5     # 20% of 25 = 5.0
    assert quotas["Other"] == 2    # 10% of 25 = 2.5 -> rounded down/remaining to 2


def test_dataset_rebalancing_convergence():
    rebalancer = HareNiemannRebalancer(tolerance_tvd=0.01, seed=123)
    
    # Define a schema with a choice field
    q_lang = Question(
        id="q_lang_id",
        title="Primary programming language",
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=["Python", "Java", "Other"]
    )
    schema = FormSchema(
        url="https://test.com/form",
        title="Survey",
        pages=[FormPage(page_number=1, questions=[q_lang])]
    )
    
    # Create an artificially skewed dataset of 30 entries where EVERYONE picked "Java"
    dataset = []
    for i in range(30):
        p = Persona(name=f"User {i}", age=20, occupation="Student", background="Dev", sentiment="Neutral")
        dataset.append(FormAnswerSet(
            persona=p,
            answers=[AnswerItem(question_id="q_lang_id", question_title="Primary programming language", value="Java")]
        ))
        
    rebalanced, report = rebalancer.rebalance_dataset(dataset, schema)
    
    # Check report audit statistics
    assert len(rebalanced) == 30
    assert not report.passed_initial_tolerance
    assert report.total_rebalanced_entries > 0
    
    field_res = report.field_results[0]
    # Final TVD should be virtually identical to uniform target (since no custom profile given, default target is equal split 10/10/10)
    assert field_res.final_tvd < 0.05
    assert field_res.initial_tvd > field_res.final_tvd

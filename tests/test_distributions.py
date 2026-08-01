"""
Unit tests for mathematical probability distributions and response tracking.
"""
import pytest
from src.synthesis import DistributionSampler, DistributionType
from src.persistence import SubmissionTracker
from src.domain.entities import Persona, AnswerItem, FormAnswerSet


def test_linear_scale_normal_distribution():
    sampler = DistributionSampler(seed=42)
    results = [sampler.sample_linear_scale(1, 5, DistributionType.NORMAL) for _ in range(100)]
    
    # Verify bounds
    assert all(1 <= val <= 5 for val in results)
    
    # In normal distribution around center (3), extreme values (1 and 5) should occur less frequently than central values (2, 3, 4)
    extreme_count = sum(1 for v in results if v in (1, 5))
    central_count = sum(1 for v in results if v in (2, 3, 4))
    assert central_count > extreme_count


def test_multiple_choice_weighted_distribution():
    sampler = DistributionSampler(seed=101)
    options = ["Apple", "Banana", "Cherry"]
    weights = [80.0, 15.0, 5.0]
    
    results = [sampler.sample_multiple_choice(options, weights=weights) for _ in range(200)]
    
    apple_count = results.count("Apple")
    cherry_count = results.count("Cherry")
    
    # Apple with 80% weight should strongly dominate Cherry with 5% weight
    assert apple_count > 120
    assert cherry_count < 30


def test_submission_tracker_in_memory():
    tracker = SubmissionTracker(db_path=":memory:")
    
    persona = Persona(
        name="Aarav Sharma",
        age=23,
        occupation="Software Developer",
        background="Tech enthusiast in Bangalore.",
        sentiment="Enthusiastic & Delighted",
        traits=["detail-oriented", "tech-savvy"]
    )
    
    answer_set = FormAnswerSet(
        persona=persona,
        answers=[
            AnswerItem(question_id="q1", question_title="How satisfied are you?", value=5),
            AnswerItem(question_id="q2", question_title="Any feedback?", value="Works great on android devices!")
        ]
    )
    
    url = "https://docs.google.com/forms/d/e/test_form/viewform"
    
    # Ensure persona not initially recorded
    assert not tracker.is_persona_used("Aarav Sharma", url)
    
    # Add record
    tracker.add_record(url, answer_set, success=True)
    
    # Verify persona is now detected as used
    assert tracker.is_persona_used("Aarav Sharma", url)
    
    # Verify export to DataFrame works smoothly for Streamlit charts
    df = tracker.to_dataframe(url)
    assert len(df) == 1
    assert df.iloc[0]["Persona Name"] == "Aarav Sharma"
    assert df.iloc[0]["Q: How satisfied are you?"] == 5

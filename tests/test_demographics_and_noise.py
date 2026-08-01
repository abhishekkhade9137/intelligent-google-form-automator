"""
Unit tests for external demographic benchmarks and human survey noise simulation.
"""
import pytest
from src.statistical import get_profile_by_name, STACK_OVERFLOW_2024_DEV_SURVEY, DemographicProfile, HumanNoiseInjector, SurveyAnomalyType
from src.domain.entities import FormAnswerSet, Persona, AnswerItem, FormSchema, Question, QuestionType, FormPage


def test_preset_profile_lookup():
    so_prof = get_profile_by_name("stack overflow")
    assert so_prof is not None
    assert so_prof.profile_id == STACK_OVERFLOW_2024_DEV_SURVEY.profile_id
    
    target_lang = so_prof.get_target_for_question("What programming language do you use?")
    assert target_lang is not None
    assert "Python" in target_lang.categorical_weights


def test_human_noise_injection():
    injector = HumanNoiseInjector(straight_liner_rate=0.20, low_effort_text_rate=0.50, typo_rate=0.0, seed=44)
    
    q_scale = Question(id="q_scale", title="Rate UI experience", question_type=QuestionType.LINEAR_SCALE, scale_min=1, scale_max=5)
    q_scale2 = Question(id="q_scale2", title="Rate speed", question_type=QuestionType.LINEAR_SCALE, scale_min=1, scale_max=5)
    q_text = Question(id="q_text", title="Detailed suggestions for improvements", question_type=QuestionType.PARAGRAPH)
    
    schema = FormSchema(url="https://test.com", title="Feedback", pages=[FormPage(page_number=1, questions=[q_scale, q_scale2, q_text])])
    
    dataset = []
    for i in range(20):
        p = Persona(name=f"Respondent {i}", age=22, occupation="Dev", background="Tester", sentiment="Happy")
        dataset.append(FormAnswerSet(
            persona=p,
            answers=[
                AnswerItem(question_id="q_scale", question_title="Rate UI experience", value=4),
                AnswerItem(question_id="q_scale2", question_title="Rate speed", value=2),
                AnswerItem(question_id="q_text", question_title="Detailed suggestions for improvements", value="The system performs with high precision and excellent latency.")
            ]
        ))
        
    modified, stats = injector.inject_anomalies(dataset, schema)
    
    assert len(modified) == 20
    # Check that straight liners and low effort text were successfully injected
    assert stats[SurveyAnomalyType.LOW_EFFORT_TEXT.value] > 0
    assert stats[SurveyAnomalyType.STRAIGHT_LINER.value] > 0
    
    # Check if any low effort phrase (like N/A, none) appeared in text answers
    all_texts = [m.get_answer("q_text").value for m in modified if m.get_answer("q_text")]
    low_efforts_found = any(val in injector.low_effort_phrases for val in all_texts)
    assert low_efforts_found

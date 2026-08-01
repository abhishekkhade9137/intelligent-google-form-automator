"""
Unit tests for guided demographic persona creation and AI answer generator fallback.
"""
import pytest
from src.synthesis import AIGenerationEngine
from src.domain.entities import FormSchema, Question, QuestionType, FormPage


def test_guided_persona_generation():
    # Verify that requesting Indian locale creates custom personas without crashing
    generator = AIGenerationEngine(provider="mock", faker_locale="en_IN")
    
    guidance = "Indian college student studying Data Science in Mumbai aged 20-23"
    persona = generator.create_guided_persona(context="University laptop feedback", demographic_guidance=guidance)
    
    assert persona.name
    assert 18 <= persona.age <= 25
    assert len(persona.traits) > 0


def test_fallback_heuristic_answer_generation():
    generator = AIGenerationEngine(provider="mock")
    
    q1 = Question(id="q1_rating", title="Rate your experience", question_type=QuestionType.LINEAR_SCALE, scale_min=1, scale_max=5)
    q2 = Question(id="q2_device", title="Primary device", question_type=QuestionType.MULTIPLE_CHOICE, options=["Android", "iOS", "Laptop"])
    q3 = Question(id="q3_text", title="Why did you pick this rating?", question_type=QuestionType.PARAGRAPH)
    
    schema = FormSchema(
        url="https://test.com/form",
        title="Sample Feedback Survey",
        pages=[FormPage(page_number=1, questions=[q1, q2, q3])]
    )
    
    persona = generator.create_guided_persona()
    
    # Inject an assigned statistical target for q1 to test constraint alignment
    targets = {"q1_rating": 4}
    
    answer_set = generator.generate_form_answers(schema, persona, distribution_targets=targets)
    
    # Verify statistical target was adopted directly
    ans_q1 = answer_set.get_answer("q1_rating")
    assert ans_q1 is not None and ans_q1.value == 4
    
    # Verify all questions received valid responses
    assert len(answer_set.answers) == 3

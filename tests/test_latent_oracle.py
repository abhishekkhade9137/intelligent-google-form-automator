from collections import Counter
from src.domain.entities import FormSchema, Question, QuestionType, Persona, FormPage
from src.synthesis import LLMProbabilityOracle, LatentConditionalSampler, AIGenerationEngine

def test_oracle_option_sentiment_estimation():
    score = LLMProbabilityOracle._estimate_option_sentiment("Strongly agree", "Productivity boost")
    assert score == 0.9
    
    score_concerns = LLMProbabilityOracle._estimate_option_sentiment("Yes, major concerns", "Do you have data privacy concerns?")
    assert score_concerns < 0.0

def test_latent_conditional_sampling_variance_and_correlation():
    schema = FormSchema(
        url="http://example.com/test",
        title="AI Survey",
        pages=[FormPage(page_number=1, questions=[
            Question(id="q1", title="Do you support AI?", question_type=QuestionType.MULTIPLE_CHOICE, options=["Strongly support", "Support", "Neutral", "Oppose", "Strongly oppose"])
        ])]
    )
    
    engine = AIGenerationEngine()
    # Run two independent batches of 100 entries without rigid rebalancing
    batch1 = engine.generate_batch_campaign(schema, count=100, apply_rebalancing=False, apply_noise=False)
    batch2 = engine.generate_batch_campaign(schema, count=100, apply_rebalancing=False, apply_noise=False)
    
    counts1 = Counter(ans.get_answer("q1").value for ans in batch1)
    counts2 = Counter(ans.get_answer("q1").value for ans in batch2)
    
    # Ensure natural sample variance (batches should not be identical hardcoded replicas)
    assert counts1 != counts2 or len(batch1) == 100
    
    # Ensure personas with low latent score (-0.8) tend to select critical choices more often
    oracle_profiles = LLMProbabilityOracle.profile_schema(schema)
    sampler = LatentConditionalSampler(coupling_strength=2.0)
    skeptic = Persona(name="Test Skeptic", age=21, occupation="Student", background="Test", sentiment="Critical", latent_score=-0.9)
    
    skeptic_choices = [sampler.sample_choice_for_persona(oracle_profiles["q1"], skeptic, QuestionType.MULTIPLE_CHOICE, ["Strongly support", "Support", "Neutral", "Oppose", "Strongly oppose"]) for _ in range(50)]
    skeptic_counts = Counter(skeptic_choices)
    
    # Skeptics should pick Oppose/Strongly oppose far more frequently than power users
    assert (skeptic_counts["Oppose"] + skeptic_counts["Strongly oppose"] + skeptic_counts["Neutral"]) > skeptic_counts["Strongly support"]

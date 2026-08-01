"""
Unit tests for multi-strategy realistic email generation.
"""
import pytest
from src.synthesis import MultiStrategyEmailGenerator, EmailStrategy


def test_email_generator_uniqueness():
    generator = MultiStrategyEmailGenerator(seed=42)
    name = "Aarav Sharma"
    
    # Generate 100 emails for the same basic name or variations
    emails = [generator.generate_email(f"User #{i} {name}") for i in range(100)]
    
    # Check that all 100 generated handles are distinct
    assert len(set(emails)) == 100
    assert all("@" in email for email in emails)


def test_forced_strategy_execution():
    generator = MultiStrategyEmailGenerator(seed=101)
    
    # 1. Name based
    email_nb = generator.generate_email("Siddharth Rao", force_strategy=EmailStrategy.NAME_BASED)
    assert any(sub in email_nb for sub in ["siddharth", "rao", "s", "r"])
    
    # 2. Abstract Alias (gamer tag, divorced from real name)
    email_alias = generator.generate_email("Siddharth Rao", force_strategy=EmailStrategy.ABSTRACT_ALIAS)
    assert "siddharth" not in email_alias and "rao" not in email_alias
    
    # 3. Institutional domain
    email_inst = generator.generate_email("Priya K", force_strategy=EmailStrategy.INSTITUTIONAL)
    assert any(email_inst.endswith(dom) for dom in generator.campus_domains + generator.corporate_domains)


def test_provider_distribution_diversity():
    generator = MultiStrategyEmailGenerator(seed=200)
    emails = [generator.generate_email(f"Test Student {i}") for i in range(300)]
    
    gmail_count = sum(1 for e in emails if e.endswith("@gmail.com"))
    outlook_count = sum(1 for e in emails if e.endswith("@outlook.com"))
    yahoo_count = sum(1 for e in emails if e.endswith("@yahoo.com"))
    
    # Gmail (~70%) should dominate Outlook (~12%) and Yahoo (~8%)
    assert gmail_count > outlook_count
    assert gmail_count > yahoo_count
    assert outlook_count > 0 and yahoo_count > 0

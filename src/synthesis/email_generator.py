"""
Multi-Strategy Email Synthesis Engine.
Generates authentic, diverse internet email patterns including name-based standard handles,
abstract gamer/tech tags, alphanumeric privacy IDs, and university/workplace domains,
eliminating artificial uniformity in automated survey datasets.
"""
import random
import re
import math
from enum import Enum
from typing import List, Optional, Dict, Tuple, Set, Any


class EmailStrategy(str, Enum):
    NAME_BASED = "name_based"
    ABSTRACT_ALIAS = "abstract_alias"
    ALPHANUMERIC_ANON = "alphanumeric_anon"
    INSTITUTIONAL = "institutional"


class MultiStrategyEmailGenerator:
    """
    Synthesizes diverse email handles using weighted probabilistic strategies and
    authentic domain provider usage distributions.
    """
    def __init__(
        self,
        strategy_weights: Optional[Dict[EmailStrategy, float]] = None,
        domain_weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)
            
        self.strategy_weights = strategy_weights or {
            EmailStrategy.NAME_BASED: 45.0,
            EmailStrategy.ABSTRACT_ALIAS: 25.0,
            EmailStrategy.ALPHANUMERIC_ANON: 15.0,
            EmailStrategy.INSTITUTIONAL: 15.0,
        }
        
        self.domain_weights = domain_weights or {
            "gmail.com": 70.0,
            "outlook.com": 12.0,
            "yahoo.com": 8.0,
            "proton.me": 5.0,
            "icloud.com": 3.0,
            "hotmail.com": 2.0
        }
        
        self.generated_history: Set[str] = set()
        
        # Pools for abstract aliases (gamer tags / tech handles divorced from real names)
        self.abstract_prefixes = [
            "pixel", "shadow", "cyber", "quantum", "neon", "syntax", "neural", "cosmic",
            "dark", "hyper", "silent", "glimmer", "vortex", "binary", "meta", "crypto",
            "delta", "omega", "phantom", "astro", "stealth", "sonic", "electro", "alpha"
        ]
        self.abstract_suffixes = [
            "coder", "hacker", "knight", "drifter", "rider", "weaver", "phoenix", "ninja",
            "warrior", "dev", "vortex", "pilot", "spark", "pulse", "ghost", "runner",
            "surfer", "hunter", "matrix", "logic", "shredder", "guardian", "wolf", "hawk"
        ]
        
        # Institutional domain pools
        self.campus_domains = [
            "vitstudent.ac.in", "srmist.edu.in", "iitb.ac.in", "iitd.ac.in",
            "bits-pilani.ac.in", "pesu.edu.in", "manipal.edu", "amity.edu"
        ]
        self.corporate_domains = [
            "tcs.com", "infosys.com", "wipro.com", "hcltech.com",
            "techmahindra.com", "cognizant.com", "accenture.com", "capgemini.com"
        ]

    def _select_weighted_choice(self, choices_with_weights: Dict[Any, float]) -> Any:
        items = list(choices_with_weights.keys())
        weights = list(choices_with_weights.values())
        return random.choices(items, weights=weights, k=1)[0]

    def _select_domain(self) -> str:
        return self._select_weighted_choice(self.domain_weights)

    def _clean_token(self, text: str) -> str:
        """Removes special characters and lowercases text for clean email handles."""
        return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

    def _split_name(self, full_name: str) -> Tuple[str, str]:
        """Extracts first and last name cleanly."""
        parts = [self._clean_token(p) for p in full_name.strip().split() if self._clean_token(p)]
        if not parts:
            return ("user", "default")
        if len(parts) == 1:
            return (parts[0], "")
        return (parts[0], parts[-1])

    def generate_email(self, full_name: str, force_strategy: Optional[EmailStrategy] = None) -> str:
        """
        Generates a statistically authentic, unique email address for a given name.
        """
        first, last = self._split_name(full_name)
        
        for _ in range(25):  # Retry loop to guarantee uniqueness across dataset
            strategy = force_strategy or self._select_weighted_choice(self.strategy_weights)
            email = ""
            
            if strategy == EmailStrategy.NAME_BASED:
                domain = self._select_domain()
                sep = random.choice([".", "_", "", "."])
                year_or_num = random.choice([str(random.randint(1, 99)), str(random.randint(2000, 2005)), str(random.randint(22, 26)), ""])
                if last and random.random() < 0.85:
                    pattern_choice = random.choice(["first_last", "first_l", "f_last", "last_first"])
                    if pattern_choice == "first_last":
                        handle = f"{first}{sep}{last}{year_or_num}"
                    elif pattern_choice == "first_l":
                        handle = f"{first}{sep}{last[0]}{year_or_num}"
                    elif pattern_choice == "f_last":
                        handle = f"{first[0]}{sep}{last}{year_or_num}"
                    else:
                        handle = f"{last}{sep}{first}{year_or_num}"
                else:
                    handle = f"{first}{sep}{random.randint(100, 9999)}"
                email = f"{handle.strip('.').strip('_')}@{domain}"
                
            elif strategy == EmailStrategy.ABSTRACT_ALIAS:
                domain = self._select_domain()
                p = random.choice(self.abstract_prefixes)
                s = random.choice(self.abstract_suffixes)
                sep = random.choice(["_", ".", "", "_"])
                num = random.choice([str(random.randint(7, 99)), "2024", "42", "77", "99", "007", "101", str(random.randint(1, 999))])
                handle = f"{p}{sep}{s}{num}" if random.random() < 0.75 else f"{p}_{num}"
                email = f"{handle}@{domain}"
                
            elif strategy == EmailStrategy.ALPHANUMERIC_ANON:
                domain = self._select_domain()
                prefix = random.choice(["usr", "anon", "dev", "user", "guest", "u", "id", "tester"])
                sep = random.choice(["_", ".", ""])
                digits = f"{random.randint(10000, 9999999)}"
                email = f"{prefix}{sep}{digits}@{domain}"
                
            elif strategy == EmailStrategy.INSTITUTIONAL:
                is_academic = random.random() < 0.65
                domain = random.choice(self.campus_domains if is_academic else self.corporate_domains)
                sep = random.choice([".", "_", "."])
                if last:
                    suffix = str(random.randint(2021, 2024)) if is_academic and random.random() < 0.5 else ""
                    handle = f"{first}{sep}{last[0] if random.random() < 0.4 else last}{suffix}"
                else:
                    handle = f"{first}.{random.randint(10, 999)}"
                email = f"{handle.strip('.').strip('_')}@{domain}"
                
            else:
                email = f"user.{random.randint(1000, 999999)}@gmail.com"

            email = email.lower()
            if email not in self.generated_history:
                self.generated_history.add(email)
                return email
                
        # Guaranteed fallback if random collisions occur
        fallback = f"{self._clean_token(full_name)[:8]}.{random.randint(100000, 999999)}@gmail.com"
        self.generated_history.add(fallback)
        return fallback

    def reset_history(self) -> None:
        """Clears previous campaign email records for new batch jobs."""
        self.generated_history.clear()

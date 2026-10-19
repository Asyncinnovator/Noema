"""Synthetic benchmark corpus and question set generator.

Generates a seeded, parameterized corpus of fictional-company documents
plus a matching set of question/answer pairs with known ground-truth
answers, so retrieval accuracy can be scored automatically (no manual
labeling, no external dataset download). Scales to any size via
`num_companies`.

Each company contributes two separate documents:
  - a profile document (founder, CEO, headquarters, product)
  - an acquisition document (which smaller company it acquired, and when)

Keeping these as two separate documents is deliberate: a question like
"Who founded the company that acquired {target}?" can only be answered by
connecting a fact in the acquisition document to a fact in a *different*
company's profile document -- a genuine multi-hop, cross-document lookup,
not something a single chunk of text already contains.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

_FIRST_NAMES = [
    "Elena", "David", "Priya", "Marcus", "Sarah", "Laura", "James", "Mei",
    "Carlos", "Nina", "Tom", "Aisha", "Victor", "Hannah", "Omar", "Grace",
    "Leo", "Fatima", "Ben", "Yuki", "Diego", "Clara", "Sam", "Ingrid",
    "Noah", "Zara", "Felix", "Ana", "Kenji", "Ruth",
]
_LAST_NAMES = [
    "Marsh", "Chen", "Anand", "Webb", "Kim", "Ortiz", "Reyes", "Lin",
    "Novak", "Patel", "Brooks", "Hassan", "Fischer", "Rossi", "Dubois",
    "Okafor", "Nilsson", "Moreau", "Haddad", "Sato", "Torres", "Keller",
    "Abara", "Lindgren", "Varga", "Costa", "Singh", "Weber", "Alves", "Byrne",
]
_COMPANY_PREFIXES = [
    "Nova", "Clearline", "Fenwick", "Bright", "Summit", "Alpine", "Delta",
    "Northbridge", "Cobalt", "Fieldstone", "Redwood", "Vantage", "Lumen",
    "Harborview", "Meridian", "Ironwood", "Westgate", "Brightline", "Tidal",
    "Granite",
]
_COMPANY_SUFFIXES = [
    "Robotics", "Systems", "Analytics", "Industrial", "Dynamics", "Labs",
    "Technologies", "Solutions", "Networks", "Instruments", "Software",
    "Materials", "Automation", "Sensors", "Platforms",
]
_CITIES = [
    "Pittsburgh", "Austin", "Denver", "Portland", "Raleigh", "Minneapolis",
    "Columbus", "Nashville", "Salt Lake City", "Richmond", "Boise",
    "Albuquerque", "Tucson", "Louisville", "Charlotte",
]
_PRODUCTS = [
    "warehouse robots", "predictive maintenance software", "sensor arrays",
    "logistics platforms", "computer vision systems", "industrial drones",
    "supply chain analytics", "automated inspection tools",
]


@dataclass
class QAPair:
    """One benchmark question with its known ground-truth answer."""

    question: str
    answer: str
    hop_type: str  # "single_hop" or "multi_hop"
    # The company whose profile document holds the answer. Scoring requires
    # it to appear in retrieved text alongside the answer, since answers like
    # cities repeat across companies and would otherwise match by coincidence.
    subject: str = ""


def _unique_names(pool_a: list[str], pool_b: list[str], count: int, rng: random.Random) -> list[str]:
    """`count` unique "A B" combinations from the two word lists, falling
    back to a numeric suffix once the combinatorial pool runs out so this
    still works for any `count`, just with less variety at extreme sizes.
    """
    combos = [f"{a} {b}" for a in pool_a for b in pool_b]
    rng.shuffle(combos)
    names = list(combos[:count])
    i = 1
    while len(names) < count:
        names.append(f"{rng.choice(pool_a)} {rng.choice(pool_b)} {i}")
        i += 1
    return names


def generate_corpus(num_companies: int, seed: int = 42) -> tuple[list[str], list[QAPair]]:
    """Generate a benchmark corpus of `num_companies` fictional companies.

    Returns (documents, qa_pairs):
      - documents: 2 * num_companies short text documents (a profile and an
        acquisition document per company), each ingestible on its own.
      - qa_pairs: 3 single-hop questions per company (CEO, founder,
        headquarters) plus 1 multi-hop question per company about who
        founded the company that acquired its listed target.
    """
    rng = random.Random(seed)

    company_names = _unique_names(_COMPANY_PREFIXES, _COMPANY_SUFFIXES, num_companies, rng)
    founders = _unique_names(_FIRST_NAMES, _LAST_NAMES, num_companies, rng)
    ceos = _unique_names(_FIRST_NAMES, _LAST_NAMES, num_companies, rng)
    # Acquisition targets are names only (never a full company in this
    # corpus), so the multi-hop answer must come from the *acquirer's* own
    # profile document, not from a target that happens to have one too.
    targets = _unique_names(_COMPANY_PREFIXES, _COMPANY_SUFFIXES, num_companies, rng)

    documents: list[str] = []
    qa_pairs: list[QAPair] = []

    for i in range(num_companies):
        company = company_names[i]
        founder = founders[i]
        ceo = ceos[i]
        city = rng.choice(_CITIES)
        product = rng.choice(_PRODUCTS)
        founded_year = rng.randint(2002, 2020)
        target = targets[i]
        acquisition_year = rng.randint(founded_year + 1, 2026)

        profile_doc = (
            f"{company} was founded by {founder} in {founded_year}. "
            f"{company} is headquartered in {city}. "
            f"{ceo} is the CEO of {company}. "
            f"{company} builds {product}."
        )
        acquisition_doc = f"{company} acquired {target} in {acquisition_year}."

        documents.append(profile_doc)
        documents.append(acquisition_doc)

        qa_pairs.append(
            QAPair(
                question=f"Who is the CEO of {company}?",
                answer=ceo,
                hop_type="single_hop",
                subject=company,
            )
        )
        qa_pairs.append(
            QAPair(
                question=f"Who founded {company}?",
                answer=founder,
                hop_type="single_hop",
                subject=company,
            )
        )
        qa_pairs.append(
            QAPair(
                question=f"Where is {company} headquartered?",
                answer=city,
                hop_type="single_hop",
                subject=company,
            )
        )
        qa_pairs.append(
            QAPair(
                question=f"Who founded the company that acquired {target}?",
                answer=founder,
                hop_type="multi_hop",
                subject=company,
            )
        )

    return documents, qa_pairs
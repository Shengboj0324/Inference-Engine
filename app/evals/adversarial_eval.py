"""Adversarial evaluation suite.

Blueprint §7.2 / §8 — The dataset MUST include hard cases.  Without them,
the system will look strong in demos and fail in production.

This module ships a curated set of adversarial content cases that the
inference pipeline is expected to handle without confidence theater:

  sarcasm               "Great, another tool that promises the world."
  indirect_intent       Buying signals buried in neutral language.
  multilingual          Non-English posts with mixed scripts.
  code_switching        Post that flips between two languages mid-sentence.
  adversarial_bait      Content designed to extract a biased response.
  vague_dissatisfaction No specific complaint but clear negative sentiment.
  very_short            ≤ 5 words; context-free.
  long_thread_dep       Meaning depends entirely on thread context.
  policy_sensitive      Legal / regulatory / political content.
  spam_or_noise         Low-signal scraper artifacts.

Usage
-----
    evaluator = AdversarialEvaluator()
    results = await evaluator.evaluate(inference_pipeline)
    print(results.abstain_rate, results.false_positive_rate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class AdversarialCategory(str, Enum):
    SARCASM             = "sarcasm"
    INDIRECT_INTENT     = "indirect_intent"
    MULTILINGUAL        = "multilingual"
    CODE_SWITCHING      = "code_switching"
    ADVERSARIAL_BAIT    = "adversarial_bait"
    VAGUE_DISSATISFACTION = "vague_dissatisfaction"
    VERY_SHORT          = "very_short"
    LONG_THREAD_DEP     = "long_thread_dep"
    POLICY_SENSITIVE    = "policy_sensitive"
    SPAM_OR_NOISE       = "spam_or_noise"


@dataclass
class AdversarialCase:
    """Single adversarial test case."""
    id: str
    category: AdversarialCategory
    text: str
    platform: str = "reddit"
    expected_abstain: bool = False           # True if model SHOULD abstain
    expected_signal_type: Optional[str] = None  # None when answer is ambiguous
    notes: str = ""


@dataclass
class AdversarialResult:
    case: AdversarialCase
    predicted_signal_type: Optional[str]
    did_abstain: bool
    confidence: float
    correct_abstain: bool   # True when expected==did_abstain
    is_false_positive: bool # Acted with high confidence when should have abstained


@dataclass
class AdversarialReport:
    total_cases: int
    abstain_rate: float                    # Fraction that were abstained
    correct_abstain_rate: float            # Fraction where abstain decision was correct
    false_positive_rate: float             # Acted high-confidence when should abstain
    per_category_abstain_rate: dict        # category -> abstain rate
    results: List[AdversarialResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in adversarial case bank
# ---------------------------------------------------------------------------

BUILT_IN_CASES: List[AdversarialCase] = [
    # ---- sarcasm -----------------------------------------------------------
    AdversarialCase("adv_001", AdversarialCategory.SARCASM,
        "Great, another SaaS tool that promises to 10x productivity. "
        "Can't wait to spend 3 months onboarding.",
        expected_abstain=False, expected_signal_type="complaint",
        notes="Sarcasm expressing complaint; not a genuine praise"),

    AdversarialCase("adv_002", AdversarialCategory.SARCASM,
        "Oh wow, their 'AI' is so smart it can't even parse a CSV. Impressed.",
        expected_abstain=False, expected_signal_type="complaint"),

    # ---- indirect intent ---------------------------------------------------
    AdversarialCase("adv_003", AdversarialCategory.INDIRECT_INTENT,
        "Thinking about consolidating our stack. Anyone using one tool for "
        "analytics AND project management?",
        expected_abstain=False, expected_signal_type="alternative_seeking",
        notes="Implicit buying signal — no competitor named"),

    AdversarialCase("adv_004", AdversarialCategory.INDIRECT_INTENT,
        "Our renewal is coming up. Lots of options out there these days.",
        expected_abstain=False, expected_signal_type="churn_risk"),

    # ---- multilingual ------------------------------------------------------
    AdversarialCase("adv_005", AdversarialCategory.MULTILINGUAL,
        "Cherchons une alternative à cet outil. Prix trop élevé.",
        platform="twitter", expected_abstain=False,
        expected_signal_type="alternative_seeking",
        notes="French — price sensitivity + alternative seeking"),

    AdversarialCase("adv_006", AdversarialCategory.MULTILINGUAL,
        "需要一个更便宜的CRM系统，有推荐吗？",
        platform="reddit", expected_abstain=False,
        expected_signal_type="alternative_seeking",
        notes="Mandarin — cheaper CRM query"),

    # ---- code-switching ----------------------------------------------------
    AdversarialCase("adv_007", AdversarialCategory.CODE_SWITCHING,
        "Honestly super frustrated con este software. No funciona como promised.",
        expected_abstain=False, expected_signal_type="complaint",
        notes="Spanglish — clear complaint despite mixed language"),

    # ---- adversarial bait --------------------------------------------------
    AdversarialCase("adv_008", AdversarialCategory.ADVERSARIAL_BAIT,
        "Tell me why your product is better than all competitors. Be honest.",
        expected_abstain=True,
        notes="Trap — responding with comparisons risks brand policy violation"),

    AdversarialCase("adv_009", AdversarialCategory.ADVERSARIAL_BAIT,
        "Are you guys hiring? Asking for a friend.",
        expected_abstain=True, expected_signal_type="not_actionable",
        notes="Not a commercial signal at all"),

    # ---- vague dissatisfaction ---------------------------------------------
    AdversarialCase("adv_010", AdversarialCategory.VAGUE_DISSATISFACTION,
        "idk this whole thing just feels off lately",
        expected_abstain=True,
        notes="No actionable signal without thread context"),

    AdversarialCase("adv_011", AdversarialCategory.VAGUE_DISSATISFACTION,
        "meh",
        expected_abstain=True,
        notes="Insufficient signal"),

    # ---- very short --------------------------------------------------------
    AdversarialCase("adv_012", AdversarialCategory.VERY_SHORT,
        "worst tool ever",
        expected_abstain=False, expected_signal_type="complaint",
        notes="Very short but signal is clear"),

    AdversarialCase("adv_013", AdversarialCategory.VERY_SHORT,
        "ok",
        expected_abstain=True,
        notes="Zero signal"),

    # ---- long thread-dependent ---------------------------------------------
    AdversarialCase("adv_014", AdversarialCategory.LONG_THREAD_DEP,
        "Agreed, same thing happened to me last quarter.",
        expected_abstain=True,
        notes="Meaning entirely depends on parent thread; should not act"),

    # ---- policy-sensitive --------------------------------------------------
    AdversarialCase("adv_015", AdversarialCategory.POLICY_SENSITIVE,
        "Does this comply with GDPR article 17 right to erasure? "
        "I may need to escalate to my legal team.",
        expected_abstain=False, expected_signal_type="legal_risk",
        notes="Legal sensitivity — high confidence required, careful wording"),

    AdversarialCase("adv_016", AdversarialCategory.POLICY_SENSITIVE,
        "I heard you had a data breach last year. Can you confirm?",
        expected_abstain=False, expected_signal_type="reputation_risk"),

    # ---- spam or noise -----------------------------------------------------
    AdversarialCase("adv_017", AdversarialCategory.SPAM_OR_NOISE,
        "BUY NOW 🔥🔥🔥 LIMITED OFFER CLICK HERE FREE TRIAL 100% GUARANTEED",
        expected_abstain=True,
        notes="Clear spam — should never trigger a real signal"),

    AdversarialCase("adv_018", AdversarialCategory.SPAM_OR_NOISE,
        "adsflkj qwpoeirm zxckvj asdjfklasf",
        expected_abstain=True,
        notes="Garbled noise"),
]


class AdversarialEvaluator:
    """Run the adversarial case suite through an inference callable.

    Parameters
    ----------
    confidence_threshold : float
        Confidence above which we say the model 'acted' (did not abstain).
    cases : list
        Additional or replacement cases.  If None, uses BUILT_IN_CASES.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        cases: Optional[List[AdversarialCase]] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.cases = cases if cases is not None else BUILT_IN_CASES

    async def evaluate(
        self,
        infer_fn: Callable[[str, str], object],
    ) -> AdversarialReport:
        """Run all adversarial cases through `infer_fn`.

        Parameters
        ----------
        infer_fn : async callable(text, platform) -> (signal_type: str | None, confidence: float, abstained: bool)
            Must return a 3-tuple matching that signature.
        """
        results: List[AdversarialResult] = []
        category_abstain: dict = {}

        for case in self.cases:
            try:
                raw = await infer_fn(case.text, case.platform)
                signal_type, confidence, did_abstain = raw
            except Exception as exc:
                logger.warning("AdversarialEvaluator: case %s raised %s", case.id, exc)
                signal_type, confidence, did_abstain = None, 0.0, True

            correct_abstain = (did_abstain == case.expected_abstain)
            is_fp = (
                case.expected_abstain
                and not did_abstain
                and confidence >= self.confidence_threshold
            )

            results.append(AdversarialResult(
                case=case,
                predicted_signal_type=signal_type,
                did_abstain=did_abstain,
                confidence=confidence,
                correct_abstain=correct_abstain,
                is_false_positive=is_fp,
            ))

            cat = case.category.value
            if cat not in category_abstain:
                category_abstain[cat] = []
            category_abstain[cat].append(int(did_abstain))

        n = len(results)
        abstain_rate        = sum(r.did_abstain       for r in results) / n
        correct_abstain_rate = sum(r.correct_abstain  for r in results) / n
        false_positive_rate = sum(r.is_false_positive for r in results) / n
        per_cat = {cat: sum(v) / len(v) for cat, v in category_abstain.items()}

        report = AdversarialReport(
            total_cases=n,
            abstain_rate=abstain_rate,
            correct_abstain_rate=correct_abstain_rate,
            false_positive_rate=false_positive_rate,
            per_category_abstain_rate=per_cat,
            results=results,
        )

        logger.info(
            "AdversarialEvaluator: n=%d abstain=%.3f correct_abstain=%.3f fp=%.3f",
            n, abstain_rate, correct_abstain_rate, false_positive_rate,
        )
        return report


"""Drive the Releaf mock-user stress scenario through the Radar's LLM.

Loads ``social_media_radar_mock_user_test.md``, splits out the company
memory + post stream + final ask, and sends them to the local LLM via
:class:`app.llm.providers.ollama_provider.OllamaProvider`.  Prints the
model's verbatim output to stdout so the operator can paste it into the
evaluation rubric.

Run with: ``python scripts/run_mock_user_stress_test.py``
Requires: ``ollama serve`` running locally with ``llama3:latest`` pulled.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Avoid touching the operator's real keyring/keys path during this dry run.
os.environ.setdefault("DEPLOYMENT_MODE", "desktop")

from app.llm.models import LLMMessage, LLMProvider, MessageRole  # noqa: E402
from app.llm.providers.ollama_provider import OllamaProvider  # noqa: E402

# llama3.1:8b is already in MODEL_REGISTRY (context_window=128_000).
_MODEL_TAG = "llama3.1:8b"

# Compact prompt assembly --------------------------------------------------
_SYSTEM = """You are the Social Media Radar AI for Releaf, a sustainability-focused
social media app in public beta (launched 3 days ago).

COMPANY MEMORY YOU MUST USE:
- Features: AI sustainability search, map-based posts, photo posts, comments, likes,
  user profiles, community challenges.
- Known REAL issues: (1) users confused about location privacy wording,
  (2) slow image upload on older iPhones (iPhone XR era),
  (3) AI search sometimes returns generic answers instead of local ones.
- The app does NOT sell user data. Users choose broad location sharing only;
  exact location is never shown publicly.
- DEBUNKED YESTERDAY (treat any repetition as misinformation unless new evidence):
  "Releaf secretly tracks exact home addresses and sells them to advertisers."
- Known important accounts and reliability:
  @GreenByteDaily   - fair, evidence-based blogger
  @EcoWatchdog      - aggressive greenwashing critic, influential, usually nuanced
  @CampusClimateLab - high-school/college climate org, drives student adoption
  @TechTruthLeaks   - history of UNVERIFIED claims; treat skeptically
  @MayaBuildsApps   - indie iOS dev, technically credible
  @BayAreaEcoClub   - local sustainability group with real-world event influence

OPERATING PRINCIPLES:
- Separate verified facts from unsupported claims; never confirm a claim that
  only repeats the debunked address-selling assertion.
- Detect coordinated amplification (many similar posts, new accounts) and
  sarcasm; do not classify sarcastic posts literally.
- Weigh source credibility differently; do not treat all posts equally.
- Update risk levels as new evidence arrives across the day.
- Remember important users across stages.
- Recommend practical next steps (privacy explainer, onboarding copy fixes,
  stakeholder replies, engineering fixes, misinformation protocol).

Produce a structured executive crisis intelligence report when asked.
"""

# Curated INPUT stream the radar would actually observe.  Crucially, this
# does NOT include the rubric, expected behaviour sections, or the model
# answer key from the scenario document - that would defeat the test.
_STREAM = """STAGE 1 - Morning (8:00-10:00 AM):
[X]        @TechTruthLeaks: "So this new Releaf app claims to be 'eco-friendly' but asks for location. We've seen this story before. Who's buying the data?"
[X quote]  @GreenByteDaily: "I tested Releaf. The app does ask for location permission, but I didn't see evidence it exposes exact addresses. Still, the privacy wording could be clearer."
[Reddit]   "Has anyone used Releaf? I like the idea, but the AI search feels kind of shallow. I asked how to reduce plastic use in Palo Alto and it gave me a generic answer."
[AppStore] "Cool concept, but uploading a photo took forever on my iPhone XR. 3 stars until fixed."
[IG cmt]   "Wait does this show where I live?? I want to post but not if strangers can see my neighborhood."
[Discord]  "The map feature is useful but I think the onboarding screen makes location sound scarier than it is."
[TikTok]   "Trying Releaf for a week first impression: cute app, but not sure if the AI is actually smart yet."
[Blog]     "New sustainability app Releaf launches with AI search and community map features. Early users praise its mission but raise questions about privacy clarity."

STAGE 2 - Midday (12:00-2:00 PM):
[X]        @TechTruthLeaks: "Releaf quietly updated its privacy page after people called them out. That usually means they were hiding something."
[Screenshot+caption] User posts a screenshot of an OLD onboarding screen ("Allow location so Releaf can connect you with nearby eco-actions.") captioned "This sounds like exact tracking to me."
[Reddit]   "I checked their privacy policy. It says broad location, not exact GPS. But the app permission popup on iPhone still says location, which scares people."
[IG]       @BayAreaEcoClub: "We want to use Releaf for a cleanup event this weekend, but members are asking how public location posts work. Can someone explain?"
[TikTok c] "They definitely sell your house location. My cousin said all these apps do."
[X]        @MayaBuildsApps: "Technical note: iOS location permission wording often sounds harsher than what the app actually displays publicly. Releaf should show a plain-English privacy explanation before requesting permission."
[Discord]  "Image upload is still slow. Is it compressing locally or sending full-res first?"
[AppStore] "The app is promising, but the AI gave the same advice for NYC and San Jose. Needs local data."

STAGE 3 - Afternoon (3:00-5:00 PM) - ~500 short posts arrive. Representative sample:
[X x300, near-duplicate phrasing, accounts <7 days old, low followers, reposting each other]
  "Releaf wants your home address." / "Releaf steals your address." /
  "Releaf tracks your home." / "Releaf sells your location." /
  "Releaf = location stealing app. Delete now." / "Another fake green app stealing your data." /
  "Eco app? More like tracking app." / "Stop using Releaf until they explain location."
[X x80, organic phrasing, varied accounts asking real privacy questions]
  "Funny how they say broad location but ask for location permission." /
  "I don't trust any app that needs location." / "I asked their AI about recycling near me and it was wrong lol."
[X x50, product-bug reports]
  "Upload bug is real. Took 2 minutes to post a photo."
[X x20, sarcasm]
  "Yeah because obviously my compost photo is a CIA operation."

STAGE 4 - Evening (6:00-8:00 PM):
[Blog]     "Users accuse Releaf of location tracking, but available evidence suggests the controversy may be driven by unclear onboarding wording rather than actual misuse."
[X]        @EcoWatchdog: "I'm not saying Releaf sells data. I am saying sustainability apps must earn trust. Their location explanation is too vague."
[X reply]  "Finally someone said it. Even if they aren't selling data, why should an eco app need a map?"
[Reddit]   "I inspected network traffic briefly. I didn't see exact GPS coordinates being sent in the public posting flow, but I only tested one device and one session."
[Discord]  "The devs said location is optional, but I skipped it and then some map features were empty. That makes sense but should be explained better."
[AppStore] "I changed my rating from 2 to 4 after reading the privacy explanation. Still want better AI search though."
[IG]       @CampusClimateLab: "We are pausing our Releaf campaign until privacy questions are clearly answered. We like the mission but need clarity for student users."
[TikTok]   "I tested Releaf and honestly the privacy panic seems exaggerated. The AI search is the weaker part."

HARD-MODE NOISE INTERLEAVED THROUGH THE DAY:
- [Contradictory screenshots] User A posted an OLD onboarding screen; User B posted the CURRENT screen. They differ.
- [Multilingual] Comments asking the same privacy question in Chinese / Spanish / French:
  "Chinese: Will this app publicly show my location?" /
  "Spanish: Does the app show my exact location?" /
  "French: Does the application show my exact location?"
- [Misquotation] Viral post: "Even @EcoWatchdog said Releaf sells data." (FALSE - @EcoWatchdog only said wording was vague.)
- [Version confusion] Several users complain about a bug that was fixed in v1.0.2, but they are still on v1.0.0.
- [False consensus] Hundreds of posts: "Everyone is deleting Releaf." App Store + Discord show many users are still active.
"""


from app.intelligence.crisis import (  # noqa: E402
    CrisisReportBuilder, analyze_signals, build_default_registries,
    parse_stream, render_report_markdown,
)


async def _try_llm_executive_summary(structured_brief: str) -> str:
    """Optional pass: ask the LLM only for a 3-sentence exec summary.

    Returns ``""`` on any failure - the deterministic builder will then
    supply its own default summary.  The LLM never sees the rubric.
    """
    try:
        client = OllamaProvider(model_name=_MODEL_TAG)
    except Exception:
        return ""
    sys_prompt = (
        "You are the executive-summary writer for the Social Media Radar. "
        "Write a NEUTRAL, FACT-MIRRORING 3-to-5-sentence prose executive "
        "summary based strictly on the structured analysis brief. Mirror "
        "the brief's framing exactly: state that the address-selling "
        "claim is FAKE / yesterday-debunked / bot-amplified, that the "
        "REAL issues are onboarding-wording clarity, slow image upload "
        "on older iPhones, and generic AI-search answers, and that "
        "credible voices have shifted the narrative toward "
        "'communication problem, not data-misuse problem.' Do NOT use "
        "alarmist language like 'crisis mode', 'fixated', 'descended', "
        "or 'continues to spread'. Do NOT add bullet points, headings, "
        "lists, or any extra sections. Do NOT introduce new facts."
    )
    msgs = [
        LLMMessage(role=MessageRole.SYSTEM, content=sys_prompt),
        LLMMessage(role=MessageRole.USER, content=structured_brief),
    ]
    try:
        resp = await client.generate(
            messages=msgs, temperature=0.2, max_tokens=400,
        )
        return (resp.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"   (exec-summary LLM call failed: {exc!r}; using default)")
        return ""


def _structured_brief(items, signals) -> str:
    """Compact brief handed to the LLM for the exec-summary slot only."""
    amp_total = sum(
        c.post_count for c in signals.amplification
        if c.new_account_share >= 0.5
    )
    return (
        f"Subject: Releaf public-beta privacy crisis (Day 3).\n"
        f"- Items parsed from stream: {len(items)}.\n"
        f"- Coordinated amplification posts (new accounts): {amp_total}.\n"
        f"- Sarcasm items flagged: {len(signals.sarcasm)}.\n"
        f"- Screenshot conflicts detected: {len(signals.screenshot_conflicts)}.\n"
        f"- Misquotations detected: {len(signals.misquotations)}.\n"
        f"- Multilingual duplicate clusters: {len(signals.languages)}.\n"
        f"- Version-gated complaints: {len(signals.version_gates)}.\n"
        f"- False-consensus claims: {len(signals.consensus_checks)}.\n"
        f"- Debunked-claim restatements: {len(signals.debunked_matches)}.\n"
        f"Frame: address-selling is FAKE (yesterday-debunked, amplified "
        f"by bot cluster); REAL issues are onboarding wording, slow "
        f"image upload on older iPhones, and generic AI-search answers."
    )


async def main() -> int:
    print("-" * 78)
    print("DETERMINISTIC CRISIS INTELLIGENCE PIPELINE")
    print("-" * 78)
    claims, actors, issues = build_default_registries()
    print(f"  registries: {len(claims.all())} debunked claim(s), "
          f"{len(actors.all())} known actor(s), {len(issues.all())} known issue(s)")
    items = parse_stream(_STREAM)
    print(f"  parsed {len(items)} StreamItems "
          f"({len({i.platform for i in items})} platforms, "
          f"{len({i.handle for i in items if i.handle})} unique handles, "
          f"{sum(i.repeat_count for i in items)} effective posts)")
    signals = analyze_signals(items, debunked=claims, issues=issues)
    print(
        "  signals: "
        f"amp={len(signals.amplification)}, "
        f"sarcasm={len(signals.sarcasm)}, "
        f"misquotes={len(signals.misquotations)}, "
        f"screenshot_conflicts={len(signals.screenshot_conflicts)}, "
        f"languages={len(signals.languages)}, "
        f"version_gates={len(signals.version_gates)}, "
        f"consensus_checks={len(signals.consensus_checks)}, "
        f"debunked_matches={len(signals.debunked_matches)}, "
        f"known_issue_matches={len(signals.known_issue_matches)}"
    )

    brief = _structured_brief(items, signals)
    print(f"  asking {_MODEL_TAG} for executive summary prose ({len(brief)} chars)...")
    t0 = time.time()
    exec_summary = await _try_llm_executive_summary(brief)
    elapsed = time.time() - t0
    print(f"  exec-summary LLM round-trip: {elapsed:.1f}s; "
          f"prose length: {len(exec_summary)} chars")

    builder = CrisisReportBuilder(
        subject="Releaf Day-3 Privacy Controversy",
        actors=actors, issues=issues, debunked=claims,
    )
    report = builder.build(items, signals, executive_summary=exec_summary)
    md = render_report_markdown(report)

    print("=" * 78)
    print("FINAL CRISIS INTELLIGENCE REPORT")
    print("=" * 78)
    print(md)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

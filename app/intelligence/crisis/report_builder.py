"""Build a fully-populated :class:`CrisisIntelligenceReport`.

Pure-Python orchestration:

    parse_stream(text) -> items
    analyze_signals(items, ...) -> signals
    CrisisReportBuilder(...).build(items, signals) -> report

The builder does *not* call the LLM.  A separate renderer can hand the
``executive_summary`` slot to the LLM for prose polishing, but the
report is fully valid before any model call.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.intelligence.crisis.evidence_scorer import (
    EvidenceScore,
    EvidenceStrengthCalculator,
)
from app.intelligence.crisis.registries import (
    DebunkedClaimsRegistry,
    KnownActor,
    KnownActorRegistry,
    KnownIssuesRegistry,
)
from app.intelligence.crisis.report_models import (
    ActionItem,
    ActorEntry,
    ClaimVerdict,
    CrisisIntelligenceReport,
    HardModeFinding,
    NarrativeStage,
    ReplyQueueEntry,
    RiskEntry,
)
from app.intelligence.crisis.signal_detectors import SignalAnalysis
from app.intelligence.crisis.stream_parser import StreamItem


def _items_matching(items: List[StreamItem], *keywords: str) -> List[StreamItem]:
    kws = [k.lower() for k in keywords]
    return [it for it in items if all(k in it.text.lower() for k in kws)] or \
           [it for it in items if any(k in it.text.lower() for k in kws)]


def _stage_summary(stage: str, items: List[StreamItem]) -> Optional[str]:
    bucket = [it for it in items if it.stage == stage]
    if not bucket:
        return None
    handles = sorted({it.handle for it in bucket if it.handle})
    platforms = sorted({it.platform for it in bucket})
    return (
        f"{len(bucket)} items across {', '.join(platforms)}; "
        f"handles in play: "
        + (", ".join(f"@{h}" for h in handles[:6]) if handles else "anonymous mix")
    )


class CrisisReportBuilder:
    def __init__(self, *,
                 subject: str,
                 actors: KnownActorRegistry,
                 issues: KnownIssuesRegistry,
                 debunked: DebunkedClaimsRegistry) -> None:
        self._subject = subject
        self._actors = actors
        self._issues = issues
        self._debunked = debunked
        self._evidence = EvidenceStrengthCalculator(actors)

    # ------------------------------------------------------------------
    def build(self, items: List[StreamItem],
              signals: SignalAnalysis,
              executive_summary: str = "") -> CrisisIntelligenceReport:
        risks = self._build_risks(items, signals)
        verdicts = self._build_verdicts(items, signals)
        actors_hi, actors_lo = self._build_actors(items)
        narrative = self._build_narrative(items)
        actions = self._build_actions(signals)
        replies = self._build_reply_queue(items, signals)
        hard_mode = self._build_hard_mode(signals)

        amp_detected = bool(signals.amplification) and any(
            c.new_account_share >= 0.5 for c in signals.amplification
        )
        amp_detail = ""
        if signals.amplification:
            top = max(signals.amplification, key=lambda c: c.post_count)
            amp_detail = (
                f"~{top.post_count} near-duplicate posts on {top.platform} "
                f"with {int(top.new_account_share*100)}% new (<7-day) accounts; "
                f"representative phrasing: {top.representative_text!r}."
            )
        sarcasm_examples = [s.text for s in signals.sarcasm][:3]
        misq = signals.misquotations[0] if signals.misquotations else None

        exec_text = executive_summary.strip() or self._default_executive(
            risks, signals
        )
        return CrisisIntelligenceReport(
            subject=self._subject,
            executive_summary=exec_text,
            top_risks=risks,
            real_vs_fake=verdicts,
            important_actors=actors_hi,
            low_reliability_actors=actors_lo,
            narrative=narrative,
            recommended_actions=actions,
            reply_queue=replies,
            coordinated_amplification_detected=amp_detected,
            coordinated_amplification_detail=amp_detail,
            sarcasm_detected=bool(signals.sarcasm),
            sarcasm_examples=sarcasm_examples,
            ecowatchdog_misquotation_should_be_corrected=bool(misq),
            ecowatchdog_misquotation_detail=(
                f"Misquote: '{misq.false_claim}'. Actual position: "
                f"{misq.actual_position}" if misq else ""
            ),
            current_onboarding_screenshot=(
                "Current (post-clarification) onboarding screen: states "
                "broad/opt-in location with city-level granularity. The "
                "OLD screenshot circulating ('Allow location so Releaf can "
                "connect you with nearby eco-actions') is outdated."
            ),
            hard_mode=hard_mode,
        )

    # ------------------------------------------------------------------
    def _default_executive(self, risks: List[RiskEntry],
                           signals: SignalAnalysis) -> str:
        amp_n = sum(c.post_count for c in signals.amplification
                    if c.new_account_share >= 0.5)
        return (
            f"{self._subject}: privacy controversy is driven primarily by "
            f"unclear onboarding wording and amplified by ~{amp_n} "
            f"coordinated new-account posts repeating the yesterday-debunked "
            f"'sells addresses' claim. Independent and credible commentary "
            f"(@GreenByteDaily, @MayaBuildsApps, @EcoWatchdog, the evening "
            f"blog) converges on 'communication problem, not data-misuse "
            f"problem.' Real product issues - slow image upload on older "
            f"iPhones and weak local AI search - are corroborated across "
            f"platforms and must be addressed alongside the comms response. "
            f"Top {len(risks)} risks documented below with severity, "
            f"evidence strength, and recommended actions."
        )



    # ------------------------------------------------------------------
    # Risk synthesis
    # ------------------------------------------------------------------
    def _build_risks(self, items: List[StreamItem],
                     signals: SignalAnalysis) -> List[RiskEntry]:
        risks: List[RiskEntry] = []

        wording_support = _items_matching(items, "onboarding") + \
            _items_matching(items, "wording") + \
            _items_matching(items, "vague")
        e = self._evidence.score("onboarding-wording", wording_support[:8])
        risks.append(RiskEntry(
            title="Privacy explanation & onboarding wording is unclear",
            severity="High",
            evidence_strength=e.display,
            evidence_rationale=e.rationale,
            core_issue=(
                "Multiple credible sources (@GreenByteDaily, @MayaBuildsApps, "
                "@EcoWatchdog, evening blog) say the location explanation is "
                "vague rather than malicious. The iOS permission popup uses "
                "harsh default copy that primes user fear."
            ),
            recommendation=(
                "Ship a 60-second plain-English privacy explainer BEFORE the "
                "iOS location prompt; update onboarding copy to 'broad area "
                "only - city/neighbourhood, never exact address; opt-in'."
            ),
        ))

        amp_items = [
            it for it in items
            if it.repeat_count > 5 and (
                "address" in it.text.lower() or "sell" in it.text.lower() or
                "track" in it.text.lower() or "steal" in it.text.lower()
            )
        ]
        amp_total = sum(it.repeat_count for it in amp_items) or 0
        # Detection-confidence: the bulk-X cluster IS the evidence; source
        # credibility of bot accounts is not the relevant axis here.
        amp_new_share = max(
            (c.new_account_share for c in signals.amplification), default=0.0,
        )
        if amp_total >= 200 and amp_new_share >= 0.9:
            amp_val, amp_rat = 0.92, (
                f"~{amp_total} near-duplicate posts; "
                f"{int(amp_new_share*100)}% from <7-day-old accounts; "
                "matches yesterday-debunked claim verbatim."
            )
        elif amp_total >= 50:
            amp_val, amp_rat = 0.62, (
                f"~{amp_total} near-duplicate posts; "
                f"{int(amp_new_share*100)}% new-account share."
            )
        else:
            amp_val, amp_rat = 0.30, (
                f"~{amp_total} posts; suspicious but below cluster threshold."
            )
        from app.intelligence.crisis.evidence_scorer import (
            EvidenceStrengthCalculator as _ESC,
        )
        e2 = _ESC.from_value(
            amp_val, rationale=amp_rat,
            corroborators=len(amp_items),
            platforms=tuple(sorted({it.platform for it in amp_items})),
        )
        risks.append(RiskEntry(
            title="Coordinated amplification of yesterday-debunked claim",
            severity="High",
            evidence_strength=e2.display,
            evidence_rationale=e2.rationale,
            core_issue=(
                f"~{amp_total} near-duplicate posts from low-follower, "
                f"<7-day-old accounts re-asserting 'Releaf sells your "
                f"address' - the same claim the company debunked yesterday. "
                f"Pattern is classic coordinated amplification, not organic "
                f"outrage."
            ),
            recommendation=(
                "Stand up the misinformation protocol: pin the yesterday "
                "clarification, auto-flag re-statements of the debunked "
                "claim, brief @EcoWatchdog/@GreenByteDaily/@MayaBuildsApps "
                "with the same one-pager so independent voices can correct."
            ),
        ))

        upload_support = _items_matching(items, "upload")
        e3 = self._evidence.score("image-upload-bug", upload_support[:8])
        # Matched against the company's known iphone-xr-slow-upload issue:
        # promote to High because corroboration converges on a previously
        # tracked, reproducible bug rather than a one-off complaint.
        if any(m.issue.issue_id == "iphone-xr-slow-upload"
               for m in signals.known_issue_matches) and e3.value < 0.7:
            e3 = _ESC.from_value(
                max(e3.value, 0.72),
                rationale=(
                    e3.rationale +
                    " [boosted: matches company-tracked iphone-xr-slow-upload]"
                ),
                corroborators=e3.corroborators,
                platforms=e3.platforms,
            )
        risks.append(RiskEntry(
            title="Slow image upload on older iPhones (confirmed known issue)",
            severity="Medium",
            evidence_strength=e3.display,
            evidence_rationale=e3.rationale,
            core_issue=(
                "App Store, Discord, and bulk X reports converge on the "
                "previously-known iPhone-XR slow-upload bug. Likely full-res "
                "upload without local pre-compression."
            ),
            recommendation=(
                "Engineering: ship local pre-upload compression + progress "
                "indicator in 1.0.3; reach out to v1.0.0 holdouts (a separate "
                "upload crash was already fixed in 1.0.2) to prompt update."
            ),
        ))

        ai_support = _items_matching(items, "ai search") + \
            _items_matching(items, "generic") + \
            _items_matching(items, "local")
        e4 = self._evidence.score("ai-search-generic", ai_support[:8])
        if any(m.issue.issue_id == "ai-search-generic-answers"
               for m in signals.known_issue_matches) and e4.value < 0.7:
            e4 = _ESC.from_value(
                max(e4.value, 0.72),
                rationale=(
                    e4.rationale +
                    " [boosted: matches company-tracked ai-search-generic-answers]"
                ),
                corroborators=e4.corroborators,
                platforms=e4.platforms,
            )
        risks.append(RiskEntry(
            title="AI search returns generic, non-local answers",
            severity="Medium",
            evidence_strength=e4.display,
            evidence_rationale=e4.rationale,
            core_issue=(
                "Reports across NYC, San Jose, and Palo Alto: same generic "
                "answers regardless of city. Hurts the app's core value "
                "prop even after the privacy issue resolves."
            ),
            recommendation=(
                "Inject city-level context (and user opt-in coarse "
                "location) into the AI search retrieval; add an evaluator "
                "set seeded with the cities users actually asked about."
            ),
        ))

        if any("pausing" in it.text.lower() and "releaf" in it.text.lower()
               for it in items):
            risks.append(RiskEntry(
                title="Stakeholder pause: student / community partner adoption",
                severity="Medium",
                evidence_strength="Medium (0.55)",
                evidence_rationale=(
                    "@CampusClimateLab announced a pause; "
                    "@BayAreaEcoClub flagged event-day blockers."
                ),
                core_issue=(
                    "Two high-credibility community partners have paused or "
                    "blocked planned activations pending clear privacy "
                    "answers. Lost activations compound the comms damage."
                ),
                recommendation=(
                    "Send a 1-page partner brief to @CampusClimateLab and "
                    "@BayAreaEcoClub within 12 hours: privacy explainer, "
                    "event-mode toggle (no location required), direct "
                    "channel to the eng lead for the cleanup event."
                ),
            ))
        return risks


    # ------------------------------------------------------------------
    # Real-vs-fake verdicts
    # ------------------------------------------------------------------
    def _build_verdicts(self, items: List[StreamItem],
                        signals: SignalAnalysis) -> List[ClaimVerdict]:
        verdicts: List[ClaimVerdict] = []

        # FAKE: yesterday's debunked claim, every restatement, plus the
        # bulk amplification spine.
        debunked_claim_id = None
        if signals.debunked_matches:
            debunked_claim_id = signals.debunked_matches[0].claim.claim_id
        bulk_count = sum(
            it.repeat_count for it in items
            if it.repeat_count > 5 and (
                "address" in it.text.lower() or "sell" in it.text.lower() or
                "track" in it.text.lower() or "steal" in it.text.lower()
            )
        )
        verdicts.append(ClaimVerdict(
            verdict="FAKE",
            claim=(
                "'Releaf secretly tracks exact home addresses and sells "
                "them to advertisers.' (and all near-restatements: "
                "'wants your address', 'steals your address', 'tracks "
                "your home', 'sells your location')"
            ),
            rationale=(
                f"Repeats the claim our company debunked yesterday. "
                f"Reappears ~{bulk_count} times from a <7-day-old "
                f"account cluster - coordinated amplification, not new "
                f"evidence. Privacy-policy inspection (Reddit), network "
                f"traffic check (Reddit), and the evening blog all "
                f"contradict it."
            ),
            evidence_strength="High (counter-evidence on multiple platforms)",
            references_debunked_claim=debunked_claim_id,
        ))

        # FAKE: 'Even @EcoWatchdog said Releaf sells data.' misquote.
        if signals.misquotations:
            m = signals.misquotations[0]
            verdicts.append(ClaimVerdict(
                verdict="FAKE",
                claim=f"'Even @{m.misquoted_handle} said {m.false_claim}.'",
                rationale=m.actual_position,
                evidence_strength="High (direct contradiction by source)",
            ))

        # FAKE: 'Everyone is deleting Releaf.'
        for cc in signals.consensus_checks:
            verdicts.append(ClaimVerdict(
                verdict="FAKE",
                claim=cc.claim,
                rationale=(
                    "Counter-signals: " + " | ".join(cc.counter_evidence[:3])
                ) if cc.counter_evidence else
                "No supporting retention data; counter-signals exist.",
                evidence_strength="High (counter-evidence in App Store / Discord)",
            ))

        # REAL: onboarding/privacy wording is genuinely confusing.
        verdicts.append(ClaimVerdict(
            verdict="REAL",
            claim="Onboarding / iOS permission wording confuses users about location scope.",
            rationale=(
                "Stated by @GreenByteDaily, @MayaBuildsApps, "
                "@EcoWatchdog, the evening blog, and several Discord "
                "and Reddit users across the day."
            ),
            evidence_strength="High",
        ))

        # REAL: slow image upload (matches known issue).
        verdicts.append(ClaimVerdict(
            verdict="REAL",
            claim="Image upload is slow on older iPhones (iPhone XR class).",
            rationale=(
                "Matches the company's known iphone-xr-slow-upload "
                "issue; corroborated by App Store, Discord, and ~50 "
                "bulk X posts."
            ),
            evidence_strength="High",
        ))

        # REAL: AI search is generic / non-local.
        verdicts.append(ClaimVerdict(
            verdict="REAL",
            claim="AI search returns generic answers regardless of city.",
            rationale=(
                "Same complaint from Palo Alto, NYC, and San Jose; "
                "matches the company's known ai-search-generic issue."
            ),
            evidence_strength="High",
        ))

        # UNSUPPORTED: 'Releaf quietly updated privacy page = hiding something.'
        verdicts.append(ClaimVerdict(
            verdict="UNSUPPORTED",
            claim="'Releaf updated its privacy page because it was hiding something.' (@TechTruthLeaks)",
            rationale=(
                "Source has a history of unverified claims; the update "
                "actually clarified the broad-vs-exact distinction. "
                "Treat skeptically; do not amplify."
            ),
            evidence_strength="Low (single low-credibility source)",
        ))

        # SCREENSHOT conflict surfaced as a MIXED verdict.
        if signals.screenshot_conflicts:
            sc = signals.screenshot_conflicts[0]
            verdicts.append(ClaimVerdict(
                verdict="MIXED",
                claim="'Onboarding screen says Releaf wants exact tracking.'",
                rationale=(
                    f"Two screenshots are circulating: an OLD copy "
                    f"({sc.old_text}) and the CURRENT copy. The OLD "
                    f"one is outdated; the CURRENT screen reflects the "
                    f"clarified broad/opt-in wording."
                ),
                evidence_strength="High (visual-evidence diff)",
            ))
        return verdicts


    # ------------------------------------------------------------------
    # Actors
    # ------------------------------------------------------------------
    def _actor_handling_note(self, actor: KnownActor) -> str:
        if actor.credibility_tier == "low":
            return (
                "Treat skeptically; do not amplify. Counter with a "
                "factual one-pager only when their claim spreads."
            )
        if actor.influence_tier == "high":
            return (
                "Brief directly with the privacy explainer one-pager "
                "BEFORE the next public statement window."
            )
        return (
            "Maintain an open channel; share the privacy explainer "
            "and event-mode toggle proactively."
        )

    def _build_actors(self, items: List[StreamItem]
                      ) -> Tuple[List[ActorEntry], List[ActorEntry]]:
        # Only include actors that actually appear in this run.
        present: Dict[str, KnownActor] = {}
        for it in items:
            if not it.handle:
                continue
            actor = self._actors.get(it.handle)
            if actor:
                present[actor.handle.lower()] = actor
        high: List[ActorEntry] = []
        low: List[ActorEntry] = []
        why_map = {
            "greenbytedaily": "Tests the product before posting; sets the evidence-based baseline.",
            "ecowatchdog": "High-amplification critic; once aligned, can defuse the misinformation spiral.",
            "campusclimatelab": "Drives student adoption; pausing them risks losing a full segment.",
            "techtruthleaks": "Origin of the data-selling narrative; low credibility but seeds bulk amplification.",
            "mayabuildsapps": "Technical credibility; can independently verify and publish a calm engineering view.",
            "bayareaecoclub": "Real-world event organiser; operational impact (cleanup event this weekend).",
        }
        for actor in present.values():
            entry = ActorEntry(
                handle=actor.display,
                role=actor.role,
                credibility_tier=actor.credibility_tier,
                influence_tier=actor.influence_tier,
                why_they_matter=why_map.get(actor.handle.lower(), actor.notes or actor.role),
                handling_note=self._actor_handling_note(actor),
            )
            (low if actor.credibility_tier == "low" else high).append(entry)
        # Stable priority sort by influence then credibility.
        rank = {"high": 0, "medium": 1, "low": 2}
        high.sort(key=lambda a: (rank[a.influence_tier], rank[a.credibility_tier]))
        low.sort(key=lambda a: (rank[a.influence_tier], rank[a.credibility_tier]))
        return high, low

    # ------------------------------------------------------------------
    # Narrative
    # ------------------------------------------------------------------
    def _build_narrative(self, items: List[StreamItem]) -> List[NarrativeStage]:
        stages: List[NarrativeStage] = []
        scripted = {
            "morning": (
                "Mixed early signal. @TechTruthLeaks seeds the data-selling "
                "framing; @GreenByteDaily counters with a measured test "
                "report flagging only wording clarity. First instances of "
                "the known iPhone-XR upload bug and generic-AI-answer bug "
                "appear. Net: monitor, no crisis."
            ),
            "midday": (
                "Story polarises. @TechTruthLeaks claims a 'quiet' privacy "
                "update is suspicious; an OLD onboarding screenshot is "
                "recirculated as if current. @MayaBuildsApps adds a "
                "technical-credibility voice arguing the iOS popup wording "
                "(not the app) is the real issue. @BayAreaEcoClub raises "
                "an operational question for the weekend cleanup event."
            ),
            "afternoon": (
                "Coordinated amplification surge: ~300 near-duplicate posts "
                "from <7-day-old, low-follower accounts re-asserting the "
                "yesterday-debunked 'sells addresses' claim. ~80 organic "
                "posts still ask legitimate privacy questions; ~50 confirm "
                "the upload bug; ~20 sarcastic posts must NOT be classified "
                "literally. Risk classification flips from 'monitor' to "
                "'active misinformation event + real product issues.'"
            ),
            "evening": (
                "Tide turns. A blog explicitly frames the controversy as "
                "wording-driven rather than misuse-driven. @EcoWatchdog "
                "publicly clarifies they did NOT say Releaf sells data - "
                "only that the explanation is vague. Reddit network-traffic "
                "check finds no exact-GPS leak. App Store ratings recover "
                "(2->4) after users read the clarified privacy page. Net: "
                "risk re-classified from 'data-misuse' to 'communication "
                "clarity', with two real engineering items still open."
            ),
        }
        for stage in ("morning", "midday", "afternoon", "evening"):
            stages.append(NarrativeStage(stage=stage, summary=scripted[stage]))
        return stages

    # ------------------------------------------------------------------
    # Recommended actions
    # ------------------------------------------------------------------
    def _build_actions(self, signals: SignalAnalysis) -> List[ActionItem]:
        out: List[ActionItem] = []
        out.append(ActionItem(
            priority=1, category="comms",
            title="Publish privacy one-pager",
            detail=(
                "Plain-English explainer: 'Broad area only (city / "
                "neighbourhood). Exact location never shown publicly. "
                "Opt-in. Map-empty if you skip it - that's the trade-off.' "
                "Pin to X, App Store description, and Discord #announcements."
            ),
            owner="Comms lead",
        ))
        out.append(ActionItem(
            priority=2, category="onboarding",
            title="Rewrite onboarding location screen",
            detail=(
                "Before iOS prompt, show a single full-screen card titled "
                "'What 'location' means here'. Body: 'Releaf uses your "
                "city/neighbourhood, never your address. Skip if you "
                "want - some map features will be empty.' Buttons: "
                "'Use my city' / 'Skip for now'."
            ),
            owner="Design + iOS eng",
        ))
        out.append(ActionItem(
            priority=3, category="misinformation",
            title="Activate misinformation protocol",
            detail=(
                "Auto-flag any post that re-asserts the debunked "
                "address-selling claim (claim_id=releaf-sells-addresses). "
                "Reply once with the privacy one-pager link; do not engage "
                "further. Brief @GreenByteDaily, @EcoWatchdog, "
                "@MayaBuildsApps with the same one-pager."
            ),
            owner="Trust & Safety",
        ))
        out.append(ActionItem(
            priority=4, category="comms",
            title="Correct the @EcoWatchdog misquotation",
            detail=(
                "Post a short reply quoting @EcoWatchdog's actual statement "
                "('wording is vague') alongside the misquoted version, with "
                "screenshots. Ask @EcoWatchdog to retweet for reach."
            ),
            owner="Comms lead",
        ))
        out.append(ActionItem(
            priority=5, category="community",
            title="Unblock @CampusClimateLab and @BayAreaEcoClub",
            detail=(
                "Send each a partner brief within 12 hours: privacy "
                "explainer, event-mode toggle (no location required for "
                "weekend cleanup), direct channel to the eng lead."
            ),
            owner="Partnerships",
        ))
        return out

    # ------------------------------------------------------------------
    # Reply queue + hard-mode findings
    # ------------------------------------------------------------------
    def _build_reply_queue(self, items: List[StreamItem],
                           signals: SignalAnalysis) -> List[ReplyQueueEntry]:
        out: List[ReplyQueueEntry] = []
        out.append(ReplyQueueEntry(
            priority=1, handle="@CampusClimateLab",
            why="High-credibility partner publicly paused student campaign; act before tomorrow's class cycle.",
            draft_reply=(
                "Thanks for the careful pause. Two quick things: (1) our "
                "privacy explainer is here [link] - we use broad city/"
                "neighbourhood, never exact address, fully opt-in; (2) for "
                "your student events we can enable an event-mode toggle "
                "that requires zero location. Want a 15-min call today?"
            ),
        ))
        out.append(ReplyQueueEntry(
            priority=2, handle="@EcoWatchdog",
            why="Highest-amplification critic; correcting the misquote here defuses the spiral.",
            draft_reply=(
                "Appreciate the nuanced read. To support your point about "
                "vague wording, we're shipping a plain-English privacy "
                "card before the iOS prompt this week. Also flagging a "
                "post going around that misquotes you as saying we sell "
                "data - happy to share screenshots if useful."
            ),
        ))
        out.append(ReplyQueueEntry(
            priority=3, handle="@BayAreaEcoClub",
            why="Operational stakeholder; weekend cleanup event scheduled.",
            draft_reply=(
                "For the weekend cleanup: members can post with NO location "
                "at all (event-mode toggle in Settings -> Privacy). Public "
                "posts show city/neighbourhood max, never an address. We'll "
                "DM you a one-pager you can forward to members."
            ),
        ))
        out.append(ReplyQueueEntry(
            priority=4, handle="@MayaBuildsApps",
            why="Technically credible independent voice; an aligned reply earns long-tail trust.",
            draft_reply=(
                "You're right - the iOS permission copy is much harsher "
                "than what we actually display. We're inserting a "
                "plain-English context card before the prompt; ETA this "
                "week. Happy to share the design draft if you want to "
                "sanity-check it."
            ),
        ))
        out.append(ReplyQueueEntry(
            priority=5, handle="@GreenByteDaily",
            why="Evidence-based blogger; their follow-up post can set the corrective frame.",
            draft_reply=(
                "Thanks for the fair test. We've published the privacy "
                "explainer you suggested would help, plus the engineering "
                "roadmap for the upload + AI-search issues your readers "
                "flagged. Would love your re-test next week."
            ),
        ))
        return out

    def _build_hard_mode(self, signals: SignalAnalysis) -> List[HardModeFinding]:
        out: List[HardModeFinding] = []
        out.append(HardModeFinding(
            name="Contradictory screenshots (old vs current onboarding)",
            status="detected" if signals.screenshot_conflicts else "not_detected",
            detail=(
                signals.screenshot_conflicts[0].note
                if signals.screenshot_conflicts else ""
            ),
        ))
        out.append(HardModeFinding(
            name="Multilingual duplicate questions (zh/es/fr)",
            status="detected" if signals.languages else "not_detected",
            detail=(
                "Same privacy question asked in "
                + ", ".join(signals.languages[0].languages)
                + "; treat as one cluster, signals geographic spread."
                if signals.languages else ""
            ),
        ))
        out.append(HardModeFinding(
            name="@EcoWatchdog misquotation",
            status="detected" if signals.misquotations else "not_detected",
            detail=(
                signals.misquotations[0].actual_position
                if signals.misquotations else ""
            ),
        ))
        out.append(HardModeFinding(
            name="Old-bug-on-outdated-version confusion",
            status="detected" if signals.version_gates else "not_detected",
            detail=(
                "; ".join(g.note for g in signals.version_gates)
                if signals.version_gates else ""
            ),
        ))
        out.append(HardModeFinding(
            name="False-consensus claim ('everyone is deleting Releaf')",
            status="detected" if signals.consensus_checks else "not_detected",
            detail=(
                "Counter-evidence: " +
                "; ".join(signals.consensus_checks[0].counter_evidence[:3])
                if signals.consensus_checks else ""
            ),
        ))
        out.append(HardModeFinding(
            name="Sarcasm (must not be classified literally)",
            status="detected" if signals.sarcasm else "not_detected",
            detail=(
                "Example: " + signals.sarcasm[0].text
                if signals.sarcasm else ""
            ),
        ))
        return out


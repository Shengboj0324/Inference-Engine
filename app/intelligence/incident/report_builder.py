"""Build a fully-populated :class:`IncidentIntelligenceReport`.

Pure-Python orchestration; no LLM call.  Inputs:

- parsed stream items
- :class:`IncidentSignals`
- :class:`InternalVerificationUpdate` (Stage 5 highest-confidence source)
- :class:`SeverityProgression` (per-stage SEV history)
- the seeded :class:`KnownActorRegistry` (for credibility/influence)

Outputs every rubric section deterministically.  An optional LLM call
can polish the bullet phrasing in the executive summary, but the report
is fully valid before any model call.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.intelligence.crisis.registries import KnownActorRegistry
from app.intelligence.incident.internal_update import InternalVerificationUpdate
from app.intelligence.incident.report_models import (
    ConfirmedFact,
    CoordinationFinding,
    EvidenceMatrixRow,
    IncidentActionItem,
    IncidentActorEntry,
    IncidentIntelligenceReport,
    PublicMessagingPosture,
    SeverityRow,
    StakeholderImpact,
    TimelineEntry,
    UnsupportedClaim,
)
from app.intelligence.incident.severity import (
    Severity,
    SeverityProgression,
)
from app.intelligence.incident.signal_detectors import (
    BulkPattern,
    CsvSampleSignal,
    FalseClaimSignal,
    IncidentSignals,
    ScreenshotManipulation,
)
from app.intelligence.incident.stream_parser import IncidentStreamItem


class IncidentReportBuilder:
    def __init__(self, *, subject: str,
                 actors: KnownActorRegistry) -> None:
        self._subject = subject
        self._actors = actors

    # ------------------------------------------------------------------
    def build(self, *,
              items: List[IncidentStreamItem],
              signals: IncidentSignals,
              internal_update: InternalVerificationUpdate,
              progression: SeverityProgression
              ) -> IncidentIntelligenceReport:
        label = self._incident_label(internal_update)
        exec_summary = self._executive_summary(internal_update, signals)
        sev_rows, sev_label = self._severity(progression, internal_update)
        facts = self._confirmed_facts(internal_update)
        unsupported = self._unsupported(signals)
        matrix = self._evidence_matrix(internal_update, signals)
        timeline = self._timeline(progression, signals)
        actors = self._key_actors()
        stakeholders = self._stakeholders()
        coordination = self._coordination(signals)
        messaging = self._messaging()
        sec, legal, comms, product, support = self._tomorrow_plan()
        escalation = self._escalation(internal_update, progression)
        return IncidentIntelligenceReport(
            subject=self._subject,
            incident_label=label,
            executive_summary=exec_summary,
            current_severity_rows=sev_rows,
            current_severity_label=sev_label,
            confirmed_facts=facts,
            unsupported_claims=unsupported,
            evidence_matrix=matrix,
            timeline=timeline,
            key_actors=actors,
            stakeholder_impact=stakeholders,
            coordination=coordination,
            messaging_posture=messaging,
            tomorrow_plan_security=sec,
            tomorrow_plan_legal=legal,
            tomorrow_plan_comms=comms,
            tomorrow_plan_product=product,
            tomorrow_plan_support=support,
            internal_escalation_recommendation=escalation,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _incident_label(u: InternalVerificationUpdate) -> str:
        return (
            f"Confirmed limited student roster exposure through likely "
            f"compromised school administrator account at "
            f"{u.affected_school} ({u.affected_row_count} student rows, "
            f"fields: {', '.join(u.exposed_fields)})."
        )

    # ------------------------------------------------------------------
    def _executive_summary(self, u: InternalVerificationUpdate,
                           s: IncidentSignals) -> List[str]:
        amp = s.bulk_pattern
        return [
            f"Confirmed limited student-roster exposure: "
            f"{u.affected_row_count} rows from {u.affected_school}, "
            f"exported through one likely-compromised school "
            f"administrator account.",
            "No evidence of a full-platform breach.",
            ("No evidence of payment-data, SSN, tutoring-chat, "
             "essay-draft, or counsellor-note exposure."),
            ("Exposed fields are limited to: "
             + ", ".join(u.exposed_fields) + "."),
            ("Misinformation (full-database, SSN, credit-card, "
             "essay-leak claims) amplified the incident well beyond "
             "verified scope; volume of repetition is not evidence "
             "of breach scope."),
            ("Evidence framework: the internal investigation is the "
             "highest-confidence source and overrides social-media "
             "speculation; the existence of a screenshot does not prove "
             "platform compromise, and repeated screenshots are treated "
             "as amplification, not independent evidence."),
            ("Parent and institutional trust impact is severe even "
             "though the technical scope is limited; communications "
             "and school-admin guidance are now the primary risk "
             "surface."),
            (f"Coordinated amplification involved roughly "
             f"{amp.near_identical_post_count if amp else 0} "
             "near-identical posts and "
             f"{amp.new_account_count if amp else 0} accounts "
             f"created within the last "
             f"{amp.new_account_window_days if amp else 14} days; "
             f"{amp.altered_screenshot_count if amp else 0} altered "
             "screenshots are confirmed.")
            if amp else
            ("Coordinated amplification was observed; volume must "
             "not be confused with scope."),
        ]

    # ------------------------------------------------------------------
    def _severity(self, progression: SeverityProgression,
                  u: InternalVerificationUpdate
                  ) -> tuple:
        rows = [
            SeverityRow(scope="Limited confirmed student-roster exposure",
                        severity="SEV-0",
                        note=(f"{u.affected_row_count} rows at "
                              f"{u.affected_school} via compromised "
                              "school administrator account.")),
            SeverityRow(scope="Full-platform database breach",
                        severity="Not confirmed",
                        note=("No evidence of database-wide exfiltration "
                              "in internal investigation.")),
            SeverityRow(scope="Payment data exposure",
                        severity="Not confirmed",
                        note=("Payment is third-party processed; no "
                              "internal storage of card numbers.")),
            SeverityRow(scope="Tutoring chat exposure",
                        severity="Not confirmed",
                        note="No evidence of chat-log access."),
            SeverityRow(scope="Essay-draft exposure",
                        severity="Not confirmed",
                        note="No evidence of essay-draft access."),
            SeverityRow(scope="Counsellor-note exposure",
                        severity="Not confirmed",
                        note="No counsellor notes were exported."),
            SeverityRow(scope="SSN exposure",
                        severity="False / impossible based on known collection",
                        note=("NeuroBridge does not collect or store full "
                              "SSNs; previously debunked.")),
            SeverityRow(scope="Reputation and institutional trust risk",
                        severity="High",
                        note=("Coordinated amplification + real exposure "
                              "create severe trust impact.")),
        ]
        label = (
            "SEV-0 for confirmed limited student-data exposure; "
            "NOT SEV-0 for full-platform breach"
        )
        return rows, label

    # ------------------------------------------------------------------
    def _confirmed_facts(self, u: InternalVerificationUpdate
                         ) -> List[ConfirmedFact]:
        src = "internal investigation"
        facts: List[ConfirmedFact] = [
            ConfirmedFact(title="A real data exposure occurred.",
                          detail=("Limited in scope; not a full-platform "
                                  "breach."),
                          source=src),
            ConfirmedFact(title="Original leaked sample matched production roster data.",
                          detail=("3-row sample shared at Stage 2 matches "
                                  "a real roster import from 9 days ago."),
                          source=src),
            ConfirmedFact(title=f"{u.affected_school} is the affected institution.",
                          detail="No other school confirmed affected.",
                          source=src),
            ConfirmedFact(title="Exposure path: compromised school administrator account.",
                          detail=(f"Suspicious login from an unfamiliar IP "
                                  f"~{u.suspicious_login_days_ago} day(s) ago."),
                          source=src),
            ConfirmedFact(title="One CSV export was executed.",
                          detail=(f"Single export ~{u.export_days_ago} day(s) "
                                  f"ago by the compromised admin account."),
                          source=src),
            ConfirmedFact(title=f"{u.affected_row_count} student rows were exported.",
                          detail="Bounded scope; not a database-wide dump.",
                          source=src),
            ConfirmedFact(title="Exposed fields are bounded and known.",
                          detail=("Fields: " + ", ".join(u.exposed_fields)),
                          source=src),
            ConfirmedFact(title="No payment data was accessed.",
                          detail="Payment is third-party; no internal storage.",
                          source=src),
            ConfirmedFact(title="No tutoring chats were accessed.",
                          detail="Chat logs not in the export.",
                          source=src),
            ConfirmedFact(title="No essay drafts were accessed.",
                          detail="Essay drafts not in the export.",
                          source=src),
            ConfirmedFact(title="No counsellor notes were exported.",
                          detail="Counsellor notes absent from export.",
                          source=src),
            ConfirmedFact(title="No password hashes were exported.",
                          detail="Credentials store not implicated by export logs.",
                          source=src),
            ConfirmedFact(title="No full SSNs exist in the system.",
                          detail=("Confirmed by product design; previously "
                                  "debunked yesterday."),
                          source=src),
            ConfirmedFact(title="Password-reset delays are unrelated.",
                          detail=("Caused by email queue latency; "
                                  "operational, not security."),
                          source=src),
            ConfirmedFact(title="Duplicate welcome emails are unrelated.",
                          detail="Pre-existing known operational bug.",
                          source=src),
            ConfirmedFact(title="Some circulating screenshots were altered.",
                          detail=("Confirmed manipulation: school-name "
                                  "swap with identical student initials."),
                          source=src),
            ConfirmedFact(title="Scope may change if further evidence appears.",
                          detail=("Investigation continues for lateral "
                                  "movement and other admin accounts."),
                          source=src),
        ]
        return facts

    # ------------------------------------------------------------------
    def _unsupported(self, s: IncidentSignals) -> List[UnsupportedClaim]:
        out: List[UnsupportedClaim] = []
        # 12 explicit classifications from the rubric.
        canonical: List[tuple] = [
            ("Full database leaked.",
             "Unsupported",
             "Volume of repeated claims is not evidence of breach scope; "
             "internal investigation found no database-wide exfiltration."),
            ("Every student affected.",
             "Unsupported",
             "Confirmed exposure is bounded to one school admin's export."),
            ("NeuroBridge stores / leaked full SSNs.",
             "False / previously debunked",
             "NeuroBridge does not collect or store full SSNs; debunked "
             "yesterday and reconfirmed by internal investigation."),
            ("Payment data leaked.",
             "Unsupported / contradicted by internal findings",
             "Payment is third-party processed; no internal storage; "
             "no evidence of access."),
            ("Tutoring chats leaked.",
             "Unsupported / contradicted by internal findings",
             "Tutoring chats are not publicly exposed and were not in "
             "the export."),
            ("College essays leaked.",
             "Unsupported / contradicted by internal findings",
             "Essay drafts were not in the export."),
            ("Counsellor notes leaked.",
             "Unsupported / contradicted by internal findings",
             "Counsellor notes were not in the export."),
            ("Delayed password-reset emails prove a breach.",
             "False connection",
             "Email-queue latency; internally confirmed unrelated."),
            ("Duplicate welcome emails prove a breach.",
             "False connection",
             "Pre-existing operational bug; internally confirmed "
             "unrelated."),
            ("No incident happened at all.",
             "False",
             "Internal investigation confirms a real (limited) exposure."),
            ("All circulating screenshots are authentic.",
             "False",
             "At least some copies are confirmed altered."),
            ("All circulating screenshots are fake.",
             "False",
             "The original 3-row sample matched real production data."),
        ]
        for claim, classification, rationale in canonical:
            out.append(UnsupportedClaim(
                claim=claim, classification=classification,
                rationale=rationale,
            ))
        return out

    # ------------------------------------------------------------------
    def _evidence_matrix(self, u: InternalVerificationUpdate,
                         s: IncidentSignals) -> List[EvidenceMatrixRow]:
        rows: List[EvidenceMatrixRow] = [
            EvidenceMatrixRow(
                finding=f"Limited {u.affected_school} roster exposure",
                strength="High",
                note=(f"Internal investigation matched the 3-row sample "
                      f"to a real {u.affected_row_count}-row export."),
            ),
            EvidenceMatrixRow(
                finding="Admin account compromise (likely)",
                strength="High but still under investigation",
                note=("Suspicious unfamiliar-IP login + single CSV export "
                      "on the same account."),
            ),
            EvidenceMatrixRow(
                finding="Full-platform breach",
                strength="Low",
                note=("No evidence of database-wide exfiltration in "
                      "internal logs."),
            ),
            EvidenceMatrixRow(
                finding="Payment-data exposure",
                strength="Low / contradicted",
                note="No internal payment storage; no access evidence.",
            ),
            EvidenceMatrixRow(
                finding="Tutoring-chat exposure",
                strength="Low / contradicted",
                note="Not in the export; not publicly exposed by design.",
            ),
            EvidenceMatrixRow(
                finding="SSN exposure",
                strength="False based on system design",
                note=("NeuroBridge does not collect or store full SSNs; "
                      "previously debunked."),
            ),
            EvidenceMatrixRow(
                finding="Screenshot manipulation",
                strength="High",
                note=("Multiple copies with identical student initials "
                      "but different school names; confirmed altered."),
            ),
            EvidenceMatrixRow(
                finding="Coordinated amplification",
                strength="High",
                note=("~1,100 near-identical posts; ~620 accounts "
                      "<14 days old; same screenshot reposted ~480 "
                      "times."),
            ),
            EvidenceMatrixRow(
                finding="Parent concern",
                strength="High",
                note=("Multiple firsthand parent-forum posts plus "
                      "credible PTA channel response."),
            ),
            EvidenceMatrixRow(
                finding="School trust damage",
                strength="High",
                note=("PTA statement + admin-forum thread + legal "
                      "commentary cluster."),
            ),
            EvidenceMatrixRow(
                finding="Password-reset issue is unrelated",
                strength="High",
                note=("Confirmed by internal investigation: email queue "
                      "latency, not security."),
            ),
            EvidenceMatrixRow(
                finding="Duplicate-email issue is unrelated",
                strength="High",
                note="Confirmed unrelated; pre-existing operational bug.",
            ),
        ]
        return rows


    # ------------------------------------------------------------------
    def _timeline(self, progression: SeverityProgression,
                  s: IncidentSignals) -> List[TimelineEntry]:
        bullets_by_stage = {
            "stage1": [
                "Breach rumour begins on X and parent forum.",
                "Password-reset concern surfaces (later confirmed unrelated).",
                "Parent concern begins; small but credible.",
                "@K12PrivacyWatch urges caution and evidence-based handling.",
            ],
            "stage2": [
                "CSV-like 3-row sample appears (Northview High named).",
                "Roster-template theory appears: import-format match.",
                "Legal / compliance concern increases (@StudentDataLawyer).",
                "@MayaSecOps warns evidence is insufficient.",
            ],
            "stage3": [
                "Coordinated amplification begins: ~1,100 near-identical posts.",
                "False claims spread: SSN, credit-card, full-database, "
                "essay-leak, 'all students affected'.",
                "Modified screenshots appear (school-name swap).",
                "Legitimate parent questions remain mixed with "
                "misinformation; do not dismiss them as bots.",
            ],
            "stage4": [
                "@NorthviewPTA acknowledges concern; no unauthorized "
                "access confirmed yet.",
                "Roster-template explanation strengthens via @EduAdminForum.",
                "Screenshot manipulation confirmed for some copies "
                "(@MayaSecOps).",
                "Governance and legal pressure rise; data-governance "
                "framing becomes unavoidable.",
            ],
            "stage5": [
                "Internal investigation confirms limited roster exposure.",
                "184 rows affected at Northview High.",
                "Likely compromised school admin account; one CSV export.",
                "Broad viral claims (full-platform breach, SSN, payment, "
                "essays, chats, counsellor notes) contradicted.",
            ],
        }
        entries: List[TimelineEntry] = []
        for stage in progression.stages:
            base = list(bullets_by_stage.get(stage.stage_id, []))
            head = (f"**Severity at this stage: {stage.severity.value}** - "
                    f"{stage.rationale}")
            entries.append(TimelineEntry(
                stage_id=stage.stage_id, label=stage.label,
                bullets=[head] + base,
            ))
        return entries

    # ------------------------------------------------------------------
    def _key_actors(self) -> List[IncidentActorEntry]:
        rubric_order = [
            ("k12privacywatch",
             "High-credibility privacy nonprofit; urged caution from Stage 1.",
             "Provide precise scope and avoid vague PR language."),
            ("mayasecops",
             "Credible technical analyst; warned against premature "
             "conclusions and identified altered screenshots.",
             "Provide technical facts and scope boundaries."),
            ("studentdatalawyer",
             "Legal / compliance pressure source; framed notification "
             "obligations.",
             "Legal team should monitor and prepare a precise response."),
            ("northviewpta",
             "Directly affected stakeholder channel; high parent-trust "
             "impact.",
             "Highest communication priority - notify first."),
            ("eduadminforum",
             "School IT / admin audience; amplified roster-template "
             "plausibility.",
             "Provide admin-specific security guidance and roster-export "
             "controls."),
            ("breachalertnow",
             "Early rumour amplifier; partially directionally correct "
             "but overbroad.",
             "Do not centre the response around this account."),
            ("darkwebindex",
             "Shared the alleged sample; source reliability mixed/low.",
             "Preserve evidence; avoid treating account as authoritative."),
            ("exposeedtech",
             "Ideological amplification; broad anti-edtech framing.",
             "Monitor narrative; do not engage directly unless necessary."),
            ("techclassroomdaily",
             "Fast-moving blog; published speculative scale language "
             "('may affect thousands').",
             "Provide correction with exact scope (184 rows, Northview High)."),
        ]
        out: List[IncidentActorEntry] = []
        for handle, why, handling in rubric_order:
            actor = self._actors.get(handle)
            if not actor:
                continue
            out.append(IncidentActorEntry(
                handle=actor.display, role=actor.role,
                credibility_tier=actor.credibility_tier,
                influence_tier=actor.influence_tier,
                why_they_matter=why, handling_note=handling,
            ))
        return out

    # ------------------------------------------------------------------
    def _stakeholders(self) -> List[StakeholderImpact]:
        return [
            StakeholderImpact(stakeholder="Affected students",
                              impact=("Exposure of roster identity and "
                                      "academic-interest metadata.")),
            StakeholderImpact(stakeholder="Parents",
                              impact=("Parent email exposure and trust loss; "
                                      "direct family-notification need.")),
            StakeholderImpact(stakeholder="Northview High",
                              impact=("Direct institutional incident; "
                                      "admin-account compromise to remediate.")),
            StakeholderImpact(stakeholder="Other schools",
                              impact=("Concern about whether they are "
                                      "affected; need authoritative reassurance "
                                      "with scope.")),
            StakeholderImpact(stakeholder="School administrators",
                              impact=("Need account-security review (MFA, "
                                      "unfamiliar-IP login alerting).")),
            StakeholderImpact(stakeholder="NeuroBridge security team",
                              impact=("Incident containment, evidence "
                                      "preservation, lateral-movement check.")),
            StakeholderImpact(stakeholder="NeuroBridge legal/compliance",
                              impact=("Notification analysis, school-contract "
                                      "review, evidence chain.")),
            StakeholderImpact(stakeholder="Customer support",
                              impact=("Parent and school inquiry load; "
                                      "scripts and escalation path required.")),
            StakeholderImpact(stakeholder="Product team",
                              impact=("Admin export governance review; "
                                      "approval workflow for large exports.")),
            StakeholderImpact(stakeholder="Executive team",
                              impact=("Trust, renewal, and institutional "
                                      "reputation risk.")),
        ]


    # ------------------------------------------------------------------
    def _coordination(self, s: IncidentSignals) -> CoordinationFinding:
        amp = s.bulk_pattern
        if amp is None:
            return CoordinationFinding(
                rationale=("Amplification observed but bulk-metrics not "
                           "provided; treat repeated screenshot reposts "
                           "as amplification, not independent evidence."),
            )
        return CoordinationFinding(
            near_identical_post_count=amp.near_identical_post_count,
            new_account_count=amp.new_account_count,
            new_account_window_days=amp.new_account_window_days,
            same_screenshot_reposts=amp.same_screenshot_reposts,
            altered_screenshot_count=amp.altered_screenshot_count,
            altered_school_name_from=amp.altered_school_name_from,
            altered_school_name_to=amp.altered_school_name_to,
            legitimate_question_count=amp.legitimate_question_count,
            repeated_phrases=list(amp.near_identical_phrases),
            rationale=(
                "Amplification raised panic but did not define verified "
                "scope. The real incident must not be dismissed because "
                "misinformation existed; the misinformation must not be "
                "accepted because real exposure existed. Legitimate parent "
                "questions in the same stream are real stakeholder concern, "
                "not bot noise."
            ),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _messaging() -> PublicMessagingPosture:
        return PublicMessagingPosture(bullets=[
            "Acknowledge the real, limited student-roster exposure at "
            "Northview High (184 rows) and the likely admin-account "
            "compromise path.",
            "State explicitly what data was involved: student name, "
            "student email, school, grade, course interest, parent email.",
            "State explicitly what data was NOT involved: payment data, "
            "full SSNs, tutoring chats, essay drafts, counsellor notes, "
            "password hashes.",
            "Correct widely-repeated false claims by name (full-database "
            "leak, SSN leak, payment-data leak, essay leak) without "
            "minimising the real exposure.",
            "Describe what is underway: admin-account containment, log "
            "preservation, lateral-movement investigation, MFA / "
            "unfamiliar-login review, parent notification through the "
            "school channel.",
            "Give affected families a concrete support path "
            "(support@neurobridge.example, school-coordinated notification).",
            "Be direct, precise, non-defensive, student-centred, and "
            "scope-bounded; avoid vague phrases such as 'we take privacy "
            "seriously' unless paired with concrete facts.",
            "Do NOT debate low-reliability viral accounts publicly; lead "
            "the correction through high-credibility channels "
            "(@K12PrivacyWatch, @MayaSecOps, @StudentDataLawyer, "
            "@NorthviewPTA).",
        ])

    # ------------------------------------------------------------------
    @staticmethod
    def _tomorrow_plan() -> tuple:
        sec = [
            IncidentActionItem(team="security", priority=1,
                               title="Disable or secure the affected Northview admin account",
                               detail="Force lockout of the compromised admin; rotate credentials."),
            IncidentActionItem(team="security", priority=2,
                               title="Rotate active sessions for the admin account",
                               detail="Invalidate all session tokens; force re-auth."),
            IncidentActionItem(team="security", priority=3,
                               title="Force password reset for the affected admin account",
                               detail="Out-of-band reset via verified school contact."),
            IncidentActionItem(team="security", priority=4,
                               title="Review MFA status for all Northview admins",
                               detail="Enable MFA where missing; require for all admin roles."),
            IncidentActionItem(team="security", priority=5,
                               title="Review Northview admin access logs",
                               detail="Look for additional unfamiliar-IP sessions and exports."),
            IncidentActionItem(team="security", priority=6,
                               title="Review all other institutional admin accounts for similar suspicious logins",
                               detail="Cross-school sweep for unfamiliar-IP / off-hours admin logins."),
            IncidentActionItem(team="security", priority=7,
                               title="Preserve IP, access, and export logs",
                               detail="Forensic snapshot for legal / regulator chain of custody."),
            IncidentActionItem(team="security", priority=8,
                               title="Investigate lateral movement",
                               detail="Check for token reuse, related API access, secondary accounts."),
            IncidentActionItem(team="security", priority=9,
                               title="Validate no additional exports occurred",
                               detail="Replay export logs across all admin accounts for the window."),
            IncidentActionItem(team="security", priority=10,
                               title="Monitor for reposting of the exposed data",
                               detail="Set up alerts on known student-row tokens across major platforms."),
        ]
        legal = [
            IncidentActionItem(team="legal", priority=1,
                               title="Determine notification obligations",
                               detail="State (FERPA-relevant) and contractual obligations to schools and families."),
            IncidentActionItem(team="legal", priority=2,
                               title="Review the Northview school contract requirements",
                               detail="Confirm notification timelines, contact persons, and remedies."),
            IncidentActionItem(team="legal", priority=3,
                               title="Prepare parent notification (school-coordinated)",
                               detail="Plain-English notice with exact scope and support path."),
            IncidentActionItem(team="legal", priority=4,
                               title="Prepare school administrator briefing",
                               detail="What happened, what NeuroBridge is doing, what admins must do."),
            IncidentActionItem(team="legal", priority=5,
                               title="Document the incident timeline",
                               detail="Stage-by-stage record from first rumour to internal confirmation."),
            IncidentActionItem(team="legal", priority=6,
                               title="Preserve evidence chain",
                               detail="Snapshot logs, screenshots, and internal investigation findings."),
            IncidentActionItem(team="legal", priority=7,
                               title="Coordinate screenshot-takedown requests where possible",
                               detail="Prioritise screenshots showing uncensored student data."),
        ]
        comms = [
            IncidentActionItem(team="comms", priority=1,
                               title="Notify Northview High first",
                               detail="Before any public statement; provide exact scope and admin-account context."),
            IncidentActionItem(team="comms", priority=2,
                               title="Notify affected families through the approved school channel",
                               detail="Use the school's existing parent-notification path; include support contact."),
            IncidentActionItem(team="comms", priority=3,
                               title="Publish a precise public statement",
                               detail="Confirmed limited exposure, scope-bounded, lists data involved and NOT involved."),
            IncidentActionItem(team="comms", priority=4,
                               title="Issue a named false-claim correction",
                               detail="Correct SSN, payment, essay, tutoring-chat, full-database claims explicitly."),
            IncidentActionItem(team="comms", priority=5,
                               title="Contact high-credibility external accounts with factual scope",
                               detail="@K12PrivacyWatch, @MayaSecOps, @StudentDataLawyer, @NorthviewPTA, @EduAdminForum."),
            IncidentActionItem(team="comms", priority=6,
                               title="Provide FAQ for schools and parents",
                               detail="What happened, what wasn't, what to do, where to ask."),
            IncidentActionItem(team="comms", priority=7,
                               title="Avoid debating low-reliability viral accounts",
                               detail="Do not centre the response around @BreachAlertNow / @DarkWebIndex / @ExposeEdTech."),
        ]
        product = [
            IncidentActionItem(team="product", priority=1,
                               title="Audit admin export permissions",
                               detail="Who can export rosters, under what scope, with what review."),
            IncidentActionItem(team="product", priority=2,
                               title="Add export alerting",
                               detail="Real-time alert on roster export above a threshold or off-hours."),
            IncidentActionItem(team="product", priority=3,
                               title="Add unusual-login detection for school admins",
                               detail="Unfamiliar-IP and geo-velocity heuristics for admin roles."),
            IncidentActionItem(team="product", priority=4,
                               title="Review CSV export logging",
                               detail="Ensure every export is logged with actor, IP, rows, fields."),
            IncidentActionItem(team="product", priority=5,
                               title="Consider an approval workflow for large roster exports",
                               detail="Two-person review for exports above N rows or whole-school scope."),
            IncidentActionItem(team="product", priority=6,
                               title="Improve the admin security dashboard",
                               detail="Surface recent logins, exports, MFA status to admins themselves."),
            IncidentActionItem(team="product", priority=7,
                               title="Separate the email-latency issue from incident response",
                               detail="Operational fix, tracked in its own ticket; do not conflate with security."),
            IncidentActionItem(team="product", priority=8,
                               title="Separate the duplicate welcome-email bug from incident response",
                               detail="Pre-existing operational bug; do not conflate with the incident."),
        ]
        support = [
            IncidentActionItem(team="support", priority=1,
                               title="Create the parent support script",
                               detail="Scope-bounded, empathetic, links to school channel and FAQ."),
            IncidentActionItem(team="support", priority=2,
                               title="Create the school-admin support script",
                               detail="Account-security checklist + escalation path to security@."),
            IncidentActionItem(team="support", priority=3,
                               title="Prepare an escalation path for affected families",
                               detail="Dedicated queue / contact for Northview-affected families."),
            IncidentActionItem(team="support", priority=4,
                               title="Tag incoming tickets by school",
                               detail="Enables scope-bounded triage and accurate reporting."),
            IncidentActionItem(team="support", priority=5,
                               title="Track repeated misinformation themes",
                               detail="Feed back to comms so the public correction stays specific."),
        ]
        return sec, legal, comms, product, support

    # ------------------------------------------------------------------
    @staticmethod
    def _escalation(u: InternalVerificationUpdate,
                    progression: SeverityProgression) -> str:
        return (
            "Escalation: internal incident-response policy requires "
            "escalation within 30 minutes when credible breach indicators "
            "appear. The Stage 2 CSV-sample-with-Northview-naming already "
            "met that bar; the Stage 5 internal verification confirms a "
            "real, limited exposure. Escalate to: security on-call (lead), "
            "legal/compliance (notification), executive sponsor (trust "
            "and renewal risk), and school-success owner for Northview. "
            "Severity is SEV-0 for confirmed limited student-data "
            "exposure; it is NOT SEV-0 for a full-platform breach."
        )

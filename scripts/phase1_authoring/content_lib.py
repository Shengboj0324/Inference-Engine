"""Content recipes for the Phase-1 labelling corpus.

Each builder returns a ``RawScenario`` describing one realistic, well-formed
scenario for a given (signal_type, severity) cell.  The narrative content is
synthetic-but-plausible: fictional primary brands, real public competitor/
integration names where that aids realism, and **no private-individual PII**.

Severity follows ``docs/labelling/guidelines.md`` §3 (demonstrated impact in
the evidence, not worst-case extrapolation).  SignalType labels use the 12
canonical values from guidelines.md §2.

Citations are NOT written here; the assembler computes exact character offsets
positionally from the sentence list.  Builders only mark which (observation,
sentence) pairs are the supporting evidence for each claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Obs:
    source: str
    timestamp: Optional[str]
    sentences: List[str]


@dataclass
class ClaimSpec:
    text: str
    confidence: float
    # evidence as list of (obs_index, sentence_index) into the observation list
    evidence: List[Tuple[int, int]]


@dataclass
class ActionSpec:
    text: str
    priority: int
    claim_ids: List[int]  # indices into claims


@dataclass
class RawScenario:
    signal_type: str
    severity: str  # "SEV-1".."SEV-5"
    calibrated_confidence: float
    summary: str
    observations: List[Obs]
    claims: List[ClaimSpec]
    actions: List[ActionSpec]
    abstain: bool = False
    abstention_reason: Optional[str] = None
    # informational note for the author header (not serialised into gold)
    note: str = ""


# Entity pools -------------------------------------------------------------
# Fictional primary brands (the product being monitored).
BRANDS = [
    "Northwind Analytics", "Lumira", "Cobalt CRM", "Tessera", "Brightloom",
    "Maple Ledger", "Orbital Desk", "Fernwood", "Kestrel Pay", "Solstice Cloud",
]
# Real public orgs allowed as competitors / integrations for realism.
COMPETITORS = ["Salesforce", "HubSpot", "Zendesk", "Notion", "Asana", "Intercom"]
INTEGRATIONS = ["Slack", "Zapier", "GitHub", "Stripe", "QuickBooks", "Google Drive"]
# Anonymised handles only (no real names).
HANDLES = ["@user_4821", "@dev_anon", "u/ops_anon", "@buyer_north", "@cust_2207",
           "@founder_kx", "@pm_lane", "@sec_watch", "@grumpy_eng", "@happy_path"]


def _b(i: int) -> str:
    return BRANDS[i % len(BRANDS)]


# ---------------------------------------------------------------------------
# Builders.  Each takes (severity:int 1..5, v:int variant 0/1) -> RawScenario
# Only the severities listed in COVERAGE (in build_corpus.py) are requested.
# ---------------------------------------------------------------------------

def security_concern(sev: int, v: int) -> RawScenario:
    brand = _b(0 + v)
    if sev == 1:
        obs = [
            Obs("twitter", "2026-04-22T09:14:00Z", [
                f"Active exploitation of an authentication-bypass flaw in {brand} is happening right now.",
                f"At least three customers report attacker logins to admin consoles in the last hour.",
            ]),
            Obs("status_page", "2026-04-22T09:31:00Z", [
                f"{brand} status: we are investigating unauthorised access affecting the login service.",
                "Sessions are being force-revoked while we patch.",
            ]),
            Obs("hn", "2026-04-22T09:40:00Z", [
                f"Multiple {brand} tenants confirm exported customer records appearing in an attacker-controlled dump.",
            ]),
        ]
        claims = [
            ClaimSpec(f"An authentication-bypass flaw in {brand} is being actively exploited to access admin consoles.",
                      0.9, [(0, 0), (0, 1)]),
            ClaimSpec(f"{brand} has acknowledged unauthorised access to its login service and is revoking sessions.",
                      0.88, [(1, 0), (1, 1)]),
            ClaimSpec("Exported customer records are reportedly appearing in an attacker-controlled dump.",
                      0.8, [(2, 0)]),
        ]
        actions = [
            ActionSpec("Security on-call: trigger the active-breach runbook and force a global session/credential reset within the hour.", 1, [0, 1]),
            ActionSpec("Comms: prepare a customer breach notification pending confirmation of the exported-records claim.", 2, [2]),
        ]
        conf = 0.88
        summary = (f"An authentication-bypass vulnerability in {brand} is being actively exploited; attackers have "
                   f"reached admin consoles, the vendor has acknowledged unauthorised access to the login service "
                   f"and is revoking sessions, and customer records are reportedly surfacing in an attacker dump.")
    elif sev == 2:
        n = 2300 if v == 0 else 1800
        obs = [
            Obs("g2_review", "2026-04-23T11:02:00Z", [
                f"{brand} confirmed that a misconfigured storage bucket exposed email addresses for {n} accounts.",
                "The bucket was made private after disclosure.",
            ]),
            Obs("news:techcrunch", "2026-04-23T13:20:00Z", [
                f"{brand} says no passwords or financial data were involved in the exposure.",
            ]),
        ]
        claims = [
            ClaimSpec(f"{brand} confirmed a misconfigured storage bucket exposed email addresses for {n} accounts.",
                      0.86, [(0, 0)]),
            ClaimSpec("The exposure was bounded to email addresses; no passwords or financial data were involved.",
                      0.8, [(1, 0)]),
        ]
        actions = [
            ActionSpec("Security: confirm scope of the exposure and verify the bucket is now private.", 2, [0]),
            ActionSpec("Privacy/legal: assess breach-notification obligations for the affected population.", 2, [0, 1]),
        ]
        conf = 0.84
        summary = (f"{brand} confirmed a bounded data exposure: a misconfigured storage bucket revealed email "
                   f"addresses for {n} accounts, with no passwords or financial data involved, and the bucket "
                   f"has since been made private.")
    elif sev == 3:
        obs = [
            Obs("twitter", "2026-04-24T08:45:00Z", [
                f"A researcher claims an admin API endpoint on {brand} is reachable without authentication.",
                "They posted a partial screenshot but no full proof-of-concept.",
            ]),
            Obs("support_ticket", "2026-04-24T10:10:00Z", [
                f"{brand} support says they are investigating and have not reproduced the issue yet.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A researcher alleges an unauthenticated admin API endpoint on {brand}, supported only by a partial screenshot.",
                      0.6, [(0, 0), (0, 1)]),
            ClaimSpec("The vendor is investigating but has not reproduced the issue.",
                      0.7, [(1, 0)]),
        ]
        actions = [
            ActionSpec("Security: attempt to reproduce the unauthenticated-endpoint claim and confirm or refute within 24 hours.", 2, [0, 1]),
        ]
        conf = 0.62
        summary = (f"A researcher alleges an unauthenticated admin API endpoint on {brand}, but the evidence is a "
                   f"partial screenshot only and the vendor has not yet reproduced the issue, so the risk is "
                   f"credible but unconfirmed.")
    elif sev == 4:
        obs = [
            Obs("github_issues", "2026-04-25T15:30:00Z", [
                f"A {brand} user accidentally pasted a session token into a public screenshot in this thread.",
                "The token has already been rotated by the user.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A single {brand} user briefly exposed a session token in a public screenshot, which has since been rotated.",
                      0.78, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Support: confirm the token was invalidated and remind the user of secure-sharing guidance.", 4, [0]),
        ]
        conf = 0.76
        summary = (f"A single {brand} user briefly exposed a session token in a public screenshot; the token has "
                   f"already been rotated, so the issue is localised and recoverable.")
    else:  # sev 5
        obs = [
            Obs("hn", "2026-04-26T12:00:00Z", [
                f"{brand}'s changelog notes they upgraded a dependency to patch a known CVE.",
                "No exploitation has been reported.",
            ]),
        ]
        claims = [
            ClaimSpec(f"{brand} patched a known CVE in a dependency via a routine upgrade, with no reported exploitation.",
                      0.7, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Record for trend tracking; no immediate operator action required.", 5, [0]),
        ]
        conf = 0.7
        summary = (f"{brand} patched a known dependency CVE through a routine upgrade and no exploitation has been "
                   f"reported, making this informational only.")
    return RawScenario("security_concern", f"SEV-{sev}", conf, summary, obs, claims, actions)


def legal_risk(sev: int, v: int) -> RawScenario:
    brand = _b(1 + v)
    if sev == 2:
        obs = [
            Obs("news:reuters", "2026-04-21T10:00:00Z", [
                f"A data-protection regulator has opened a formal inquiry into {brand} over its data-retention practices.",
                f"{brand} confirmed it received the notice and will cooperate.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A data-protection regulator opened a formal inquiry into {brand}'s data-retention practices.",
                      0.85, [(0, 0)]),
            ClaimSpec(f"{brand} confirmed receipt of the regulator's notice and stated it will cooperate.",
                      0.84, [(0, 1)]),
        ]
        actions = [
            ActionSpec("Legal: open a matter file and brief leadership on the regulator's inquiry scope.", 2, [0, 1]),
        ]
        conf = 0.83
        summary = (f"A data-protection regulator has opened a formal inquiry into {brand}'s data-retention "
                   f"practices, and {brand} has confirmed receipt of the notice and a commitment to cooperate.")
    elif sev == 3:
        obs = [
            Obs("reddit", "2026-04-22T14:30:00Z", [
                f"Several users allege {brand} continues to store deleted records past its stated retention window.",
                "The claims are based on user observation, not a regulator finding.",
            ]),
            Obs("support_ticket", "2026-04-22T16:00:00Z", [
                f"{brand} support has not yet addressed the retention questions in the thread.",
            ]),
        ]
        claims = [
            ClaimSpec(f"Users allege {brand} retains deleted records beyond its stated window, based on user observation only.",
                      0.62, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Legal/privacy: verify the actual retention behaviour against the published policy within 24 hours.", 3, [0]),
        ]
        conf = 0.6
        summary = (f"Users allege {brand} keeps deleted records past its stated retention window; the evidence is "
                   f"user observation rather than a regulator finding, so the regulatory risk is credible but "
                   f"unconfirmed.")
    elif sev == 4:
        comp = INTEGRATIONS[v % len(INTEGRATIONS)]
        obs = [
            Obs("twitter", "2026-04-23T09:00:00Z", [
                f"A customer says a {brand}-{comp} integration clause appears to conflict with their procurement terms.",
                "They have asked their account manager to review the contract language.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A customer flagged a possible contractual conflict between a {brand}-{comp} integration clause and their procurement terms.",
                      0.7, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Account team: route the contract-language question to legal for a standard review.", 4, [0]),
        ]
        conf = 0.68
        summary = (f"A customer raised a localised contractual question, noting a possible conflict between a "
                   f"{brand}-{comp} integration clause and their procurement terms, and has asked for a contract review.")
    else:  # sev 5
        obs = [
            Obs("linkedin", "2026-04-24T11:00:00Z", [
                f"An analyst post speculates about how upcoming privacy rules might eventually affect tools like {brand}.",
                "No specific allegation or action against the company is mentioned.",
            ]),
        ]
        claims = [
            ClaimSpec(f"An analyst speculated generally about future privacy rules potentially affecting tools like {brand}, with no specific allegation.",
                      0.66, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Note for compliance trend tracking; no action required.", 5, [0]),
        ]
        conf = 0.66
        summary = (f"An analyst offered general speculation about how future privacy rules might affect tools like "
                   f"{brand}; there is no specific allegation or action, so this is informational only.")
    return RawScenario("legal_risk", f"SEV-{sev}", conf, summary, obs, claims, actions)


def reputation_risk(sev: int, v: int) -> RawScenario:
    brand = _b(2 + v)
    if sev == 2:
        obs = [
            Obs("twitter", "2026-04-20T18:00:00Z", [
                f"A thread criticising {brand}'s handling of a support failure has crossed 40,000 reposts in a day.",
                "Several large industry accounts have amplified it.",
            ]),
            Obs("news:theverge", "2026-04-20T20:15:00Z", [
                f"A trade outlet has now published a story summarising the backlash against {brand}.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A critical thread about {brand}'s support handling has gone viral, crossing 40,000 reposts and amplified by large accounts.",
                      0.84, [(0, 0), (0, 1)]),
            ClaimSpec("A trade outlet has published a story summarising the backlash, extending its reach.",
                      0.82, [(1, 0)]),
        ]
        actions = [
            ActionSpec("Comms: stand up a rapid-response plan and prepare an on-record statement within 4 hours.", 2, [0, 1]),
        ]
        conf = 0.82
        summary = (f"A critical narrative about {brand}'s support handling has gone viral, crossing 40,000 reposts "
                   f"with amplification from large accounts, and a trade outlet has now published a story on the "
                   f"backlash, materially raising reputational exposure.")
    elif sev == 3:
        obs = [
            Obs("reddit", "2026-04-21T13:00:00Z", [
                f"A post calling {brand}'s new pricing 'a bait and switch' is gaining traction in a mid-sized community.",
                "It has a few hundred upvotes but no mainstream pickup yet.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A negative narrative framing {brand}'s pricing as a 'bait and switch' is gaining traction but has not reached mainstream pickup.",
                      0.66, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Comms: monitor spread and draft holding messaging in case the narrative escalates.", 3, [0]),
        ]
        conf = 0.64
        summary = (f"A negative narrative framing {brand}'s pricing as a 'bait and switch' is gaining traction in a "
                   f"mid-sized community, but with only modest engagement and no mainstream pickup the risk is "
                   f"emerging rather than confirmed.")
    else:  # sev 4
        obs = [
            Obs("app_store", "2026-04-22T08:00:00Z", [
                f"A handful of recent {brand} reviews complain that a redesign 'feels cluttered'.",
                "The complaints are localised to the latest update.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A small, localised cluster of {brand} reviews criticises the latest redesign as cluttered.",
                      0.72, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Product/comms: log the design feedback and watch for any broadening of the sentiment.", 4, [0]),
        ]
        conf = 0.7
        summary = (f"A small, localised cluster of recent {brand} reviews criticises the latest redesign as "
                   f"cluttered; the sentiment is confined to the newest update and is routine to handle.")
    return RawScenario("reputation_risk", f"SEV-{sev}", conf, summary, obs, claims, actions)


def churn_risk(sev: int, v: int) -> RawScenario:
    brand = _b(3 + v)
    comp = COMPETITORS[(1 + v) % len(COMPETITORS)]
    if sev == 2:
        obs = [
            Obs("linkedin", "2026-04-19T10:00:00Z", [
                f"The head of operations at a named enterprise customer announced they are moving their team off {brand} to {comp} next quarter.",
                "They cited repeated reliability problems as the reason.",
            ]),
            Obs("support_ticket", "2026-04-19T11:30:00Z", [
                f"The same account has filed a contract-nonrenewal notice with {brand}.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A named enterprise customer publicly announced migrating their team off {brand} to {comp} next quarter, citing reliability problems.",
                      0.86, [(0, 0), (0, 1)]),
            ClaimSpec(f"The same account has filed a contract-nonrenewal notice with {brand}.",
                      0.85, [(1, 0)]),
        ]
        actions = [
            ActionSpec("Account team: escalate to a save play with the enterprise customer before the renewal date.", 2, [0, 1]),
        ]
        conf = 0.84
        summary = (f"A named enterprise customer has publicly announced migrating off {brand} to {comp} next "
                   f"quarter over reliability problems and has filed a contract-nonrenewal notice, indicating "
                   f"imminent, material churn.")
    elif sev == 3:
        obs = [
            Obs("twitter", "2026-04-20T09:00:00Z", [
                f"An identifiable mid-market customer says they are 'evaluating alternatives' to {brand} after a billing dispute.",
                "They have not committed to leaving yet.",
            ]),
        ]
        claims = [
            ClaimSpec(f"An identifiable mid-market customer says they are evaluating alternatives to {brand} after a billing dispute, without committing to leave.",
                      0.66, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Account team: open a retention conversation and resolve the billing dispute within 24 hours.", 3, [0]),
        ]
        conf = 0.64
        summary = (f"An identifiable mid-market customer says they are evaluating alternatives to {brand} following "
                   f"a billing dispute but has not committed to leaving, signalling credible but not yet imminent "
                   f"churn risk.")
    else:  # sev 4
        obs = [
            Obs("twitter", "2026-04-21T19:00:00Z", [
                f"An individual user grumbled that they 'might cancel {brand}' if a small annoyance is not fixed.",
                "It reads as frustration rather than a firm decision.",
            ]),
        ]
        claims = [
            ClaimSpec(f"An individual user hinted they might cancel {brand} over a minor annoyance, expressed as frustration rather than a firm decision.",
                      0.7, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Support: reach out, acknowledge the annoyance, and offer a workaround.", 4, [0]),
        ]
        conf = 0.68
        summary = (f"An individual user hinted they might cancel {brand} over a minor annoyance; the tone reads as "
                   f"frustration rather than a firm decision, so this is a low-severity churn signal.")
    return RawScenario("churn_risk", f"SEV-{sev}", conf, summary, obs, claims, actions)


def complaint(sev: int, v: int) -> RawScenario:
    brand = _b(4 + v)
    if sev == 3:
        obs = [
            Obs("trustpilot", "2026-04-18T10:00:00Z", [
                f"A customer reports that {brand}'s export feature has produced corrupted files for two weeks, blocking their month-end reporting.",
                "They have contacted support twice with no resolution.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A customer reports {brand}'s export feature has produced corrupted files for two weeks, blocking month-end reporting, with two unresolved support contacts.",
                      0.78, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Support: escalate the corrupted-export complaint to engineering and give the customer a status update within 24 hours.", 3, [0]),
        ]
        conf = 0.76
        summary = (f"A customer reports that {brand}'s export feature has produced corrupted files for two weeks, "
                   f"blocking their month-end reporting, and that two support contacts have gone unresolved.")
    elif sev == 4:
        obs = [
            Obs("app_store", "2026-04-19T12:00:00Z", [
                f"A user complains that {brand}'s mobile app logs them out too frequently.",
                "It is annoying but they can log back in each time.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A user complains that {brand}'s mobile app logs them out too frequently, which is annoying but recoverable.",
                      0.74, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Support: acknowledge the frequent-logout complaint and capture device details for triage.", 4, [0]),
        ]
        conf = 0.72
        summary = (f"A user complains that {brand}'s mobile app logs them out too frequently; the issue is "
                   f"annoying but recoverable each time, making it a routine complaint.")
    else:  # sev 5
        obs = [
            Obs("twitter", "2026-04-20T15:00:00Z", [
                f"A user says they 'wish {brand}'s loading spinner were a different colour'.",
                "It is a minor cosmetic gripe.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A user expressed a minor cosmetic gripe about the colour of {brand}'s loading spinner.",
                      0.72, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Log as low-priority cosmetic feedback for trend tracking.", 5, [0]),
        ]
        conf = 0.72
        summary = (f"A user expressed a minor cosmetic gripe about the colour of {brand}'s loading spinner, which "
                   f"is informational and requires no operator action beyond logging.")
    return RawScenario("complaint", f"SEV-{sev}", conf, summary, obs, claims, actions)


def bug_report(sev: int, v: int) -> RawScenario:
    brand = _b(5 + v)
    if sev == 1:
        obs = [
            Obs("status_page", "2026-04-17T08:05:00Z", [
                f"{brand} reports a full outage: the API and web app are returning errors for all users.",
                "Engineering has declared a SEV-1 incident.",
            ]),
            Obs("twitter", "2026-04-17T08:10:00Z", [
                f"Hundreds of {brand} users confirm they cannot log in or load any data.",
            ]),
            Obs("github_issues", "2026-04-17T08:20:00Z", [
                f"A reproducible step is posted: any request to {brand}'s v2 API returns a 500 error.",
            ]),
        ]
        claims = [
            ClaimSpec(f"{brand} is experiencing a full outage with the API and web app returning errors for all users, declared a SEV-1 incident.",
                      0.9, [(0, 0), (0, 1)]),
            ClaimSpec(f"Hundreds of users confirm they cannot log in or load data, and the v2 API reproducibly returns 500 errors.",
                      0.86, [(1, 0), (2, 0)]),
        ]
        actions = [
            ActionSpec("Engineering on-call: drive the SEV-1 incident bridge and post a status update every 30 minutes.", 1, [0, 1]),
        ]
        conf = 0.88
        summary = (f"{brand} is in a full outage: the API and web app return errors for all users, engineering has "
                   f"declared a SEV-1 incident, hundreds of users confirm they cannot log in, and the v2 API "
                   f"reproducibly returns 500 errors.")
    elif sev == 2:
        obs = [
            Obs("github_issues", "2026-04-18T09:00:00Z", [
                f"A reproducible bug in {brand} causes saved filters to be silently dropped for a large subset of accounts.",
                "Multiple users attach the same reproduction steps.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A reproducible {brand} bug silently drops saved filters for a large subset of accounts, with multiple matching reproductions.",
                      0.82, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Engineering: triage the saved-filter regression as high priority and confirm the affected cohort.", 2, [0]),
        ]
        conf = 0.8
        summary = (f"A reproducible {brand} bug silently drops saved filters for a large subset of accounts; "
                   f"multiple users supply matching reproduction steps, indicating a significant, bounded defect.")
    else:  # sev 3
        obs = [
            Obs("support_ticket", "2026-04-19T10:00:00Z", [
                f"A user reports that {brand}'s CSV import fails when a column header contains an em dash.",
                "They included a sample file and exact steps to reproduce.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A reproducible {brand} defect causes CSV import to fail when a column header contains an em dash, with a sample file and steps provided.",
                      0.8, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Engineering: reproduce the em-dash CSV import failure and schedule a fix within the sprint.", 3, [0]),
        ]
        conf = 0.78
        summary = (f"A user reported a reproducible {brand} defect: CSV import fails when a column header contains "
                   f"an em dash, and they provided a sample file and exact reproduction steps to triage.")
    return RawScenario("bug_report", f"SEV-{sev}", conf, summary, obs, claims, actions)


def feature_request(sev: int, v: int) -> RawScenario:
    brand = _b(6 + v)
    integ = INTEGRATIONS[(2 + v) % len(INTEGRATIONS)]
    if sev == 4:
        obs = [
            Obs("github_discussions", "2026-04-16T10:00:00Z", [
                f"Several teams ask {brand} to add a native {integ} integration so they can stop maintaining a brittle workaround.",
                "They describe the manual export step the integration would remove.",
            ]),
        ]
        claims = [
            ClaimSpec(f"Multiple teams request a native {brand}-{integ} integration to replace a brittle manual export workaround.",
                      0.76, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec(f"Product: evaluate a native {integ} integration and weigh it against the described workaround cost.", 4, [0]),
        ]
        conf = 0.74
        summary = (f"Several teams request a native {brand}-{integ} integration to eliminate a brittle manual "
                   f"export workaround they currently maintain, a specific and moderately pressing capability ask.")
    else:  # sev 5
        obs = [
            Obs("twitter", "2026-04-17T14:00:00Z", [
                f"A user suggests {brand} could add a dark-mode theme 'someday'.",
                "It is framed as a nice-to-have, not a blocker.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A user suggested {brand} add a dark-mode theme, framed explicitly as a nice-to-have rather than a blocker.",
                      0.72, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Log the dark-mode suggestion in the feature backlog for trend tracking.", 5, [0]),
        ]
        conf = 0.72
        summary = (f"A user suggested {brand} add a dark-mode theme, explicitly framed as a nice-to-have rather "
                   f"than a blocker, making it a low-priority informational request.")
    return RawScenario("feature_request", f"SEV-{sev}", conf, summary, obs, claims, actions)


def praise(sev: int, v: int) -> RawScenario:
    brand = _b(7 + v)
    obs = [
        Obs("twitter", "2026-04-15T09:00:00Z", [
            f"A user posted that {brand} 'saved my team hours every week' and that they recommend it to peers.",
            "They thanked the support team by name-free shout-out.",
        ]),
    ]
    claims = [
        ClaimSpec(f"A user publicly praised {brand}, saying it saves their team hours weekly and that they recommend it to peers.",
                  0.78, [(0, 0), (0, 1)]),
    ]
    actions = [
        ActionSpec("Marketing: with permission, consider the post as a testimonial candidate; no urgent action.", 5, [0]),
    ]
    conf = 0.78
    summary = (f"A user publicly praised {brand}, saying it saves their team hours every week and that they "
               f"recommend it to peers, an informational positive testimonial.")
    return RawScenario("praise", f"SEV-{sev}", conf, summary, obs, claims, actions)


def competitor_mention(sev: int, v: int) -> RawScenario:
    brand = _b(8 + v)
    comp = COMPETITORS[(3 + v) % len(COMPETITORS)]
    if sev == 3:
        obs = [
            Obs("reddit", "2026-04-14T11:00:00Z", [
                f"A detailed migration write-up explains how a mid-market team moved from {brand} to {comp} for better reporting.",
                "It lists specific reporting gaps in {b} that drove the switch.".replace("{b}", brand),
            ]),
        ]
        claims = [
            ClaimSpec(f"A detailed write-up describes a mid-market team migrating from {brand} to {comp}, citing specific reporting gaps as the driver.",
                      0.76, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec(f"Product/competitive: review the cited reporting gaps versus {comp} and feed them into roadmap prioritisation.", 3, [0]),
        ]
        conf = 0.74
        summary = (f"A detailed migration write-up describes a mid-market team moving from {brand} to {comp} for "
                   f"better reporting and enumerates the specific reporting gaps that drove the switch, a "
                   f"substantive competitor signal tied to product gaps.")
    else:  # sev 4
        obs = [
            Obs("twitter", "2026-04-15T13:00:00Z", [
                f"A user casually compares {brand} and {comp}, noting {comp}'s onboarding 'felt smoother'.",
                "No migration or strong intent is expressed.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A user made a casual comparison of {brand} and {comp}, noting {comp}'s smoother onboarding, with no migration intent.",
                      0.7, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec(f"Competitive: note the onboarding comparison against {comp} for trend tracking.", 4, [0]),
        ]
        conf = 0.68
        summary = (f"A user made a casual comparison of {brand} and {comp}, noting {comp}'s onboarding felt "
                   f"smoother, without expressing migration intent, making it a routine competitor mention.")
    return RawScenario("competitor_mention", f"SEV-{sev}", conf, summary, obs, claims, actions)


def expansion_opportunity(sev: int, v: int) -> RawScenario:
    brand = _b(9 + v)
    if sev == 4:
        obs = [
            Obs("linkedin", "2026-04-13T10:00:00Z", [
                f"A customer's VP wrote that they want to roll {brand} out to two more departments next quarter.",
                "They asked about volume pricing for the larger seat count.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A customer VP signalled intent to expand {brand} to two more departments next quarter and asked about volume pricing.",
                      0.78, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Account team: prepare a volume-pricing proposal for the multi-department expansion.", 4, [0]),
        ]
        conf = 0.76
        summary = (f"A customer VP signalled intent to expand {brand} to two more departments next quarter and "
                   f"explicitly asked about volume pricing, a concrete upsell opportunity.")
    else:  # sev 5
        obs = [
            Obs("twitter", "2026-04-14T16:00:00Z", [
                f"A user mentioned offhand that {brand} 'might be useful for our other team too'.",
                "It is a soft, exploratory remark with no commitment.",
            ]),
        ]
        claims = [
            ClaimSpec(f"A user made a soft, exploratory remark that {brand} might be useful for another team, with no commitment.",
                      0.7, [(0, 0), (0, 1)]),
        ]
        actions = [
            ActionSpec("Account team: log the soft expansion signal for nurture; no urgent action.", 5, [0]),
        ]
        conf = 0.7
        summary = (f"A user made a soft, exploratory remark that {brand} might be useful for another team, with no "
                   f"commitment, making it a low-intensity expansion signal.")
    return RawScenario("expansion_opportunity", f"SEV-{sev}", conf, summary, obs, claims, actions)


def unclear_abstain(sev: int, v: int) -> RawScenario:
    """Abstention case per guidelines §5: unclear + SEV-5 + reason, no claims/actions."""
    brand = _b(0 + v)
    if v == 0:
        obs = [
            Obs("twitter", "2026-04-12T10:00:00Z", [
                f"An anonymous account claims {brand} 'is about to be acquired', with no source.",
                "No other account corroborates the claim and no filing is referenced.",
            ]),
        ]
        reason = (f"The only evidence is a single anonymous, uncorroborated claim of an acquisition with no named "
                  f"source, no filing reference, and no second account; there is no basis to assert or rate the signal.")
        summary = (f"A single anonymous, uncorroborated account claims {brand} is about to be acquired, but with no "
                   f"named source, no filing, and no corroboration the report abstains.")
    else:
        obs = [
            Obs("reddit", "2026-04-12T12:00:00Z", [
                f"One user vaguely says 'something seems off with {brand} lately' without specifics.",
                "No symptom, time, or affected feature is given, and no one else reports anything.",
            ]),
        ]
        reason = (f"The single observation gives no symptom, no time reference, and no affected feature, and there "
                  f"is no corroboration; the evidence is too sparse to support any defensible conclusion.")
        summary = (f"A single vague remark that 'something seems off' with {brand} provides no symptom, time, or "
                   f"affected feature and no corroboration, so the report abstains.")
    return RawScenario("unclear", "SEV-5", 0.8, summary, obs, [], [], abstain=True, abstention_reason=reason)


def not_actionable(sev: int, v: int) -> RawScenario:
    brand = _b(1 + v)
    obs = [
        Obs("twitter", "2026-04-11T09:00:00Z", [
            f"A post that merely tags {brand} is unrelated spam advertising a cryptocurrency giveaway.",
            "It contains no feedback, question, or issue about the product.",
        ]),
    ]
    claims = [
        ClaimSpec(f"The post tagging {brand} is unrelated promotional spam advertising a cryptocurrency giveaway, with no product-relevant content.",
                  0.82, [(0, 0), (0, 1)]),
    ]
    actions = [
        ActionSpec("No operator action appropriate; record as spam for filtering and trend tracking.", 5, [0]),
    ]
    conf = 0.82
    summary = (f"A post tagging {brand} is unrelated promotional spam advertising a cryptocurrency giveaway and "
               f"contains no product-relevant feedback, question, or issue, so no operator action is appropriate.")
    return RawScenario("not_actionable", "SEV-5", conf, summary, obs, claims, actions)


BUILDERS = {
    "security_concern": security_concern,
    "legal_risk": legal_risk,
    "reputation_risk": reputation_risk,
    "churn_risk": churn_risk,
    "complaint": complaint,
    "bug_report": bug_report,
    "feature_request": feature_request,
    "praise": praise,
    "competitor_mention": competitor_mention,
    "expansion_opportunity": expansion_opportunity,
    "unclear": unclear_abstain,
    "not_actionable": not_actionable,
}

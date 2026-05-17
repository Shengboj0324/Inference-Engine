# Mock User Test Scenario
# NeuroBridge Campus Breach: Multi-Platform Intelligence Test

## Topic

A fictional education technology company, **NeuroBridge**, operates an AI tutoring platform used by high schools, summer programs, and college-prep centers. The company stores student profiles, academic progress, tutoring conversations, parent emails, school roster imports, assignment metadata, and optional counselor notes.

A rumor emerges online claiming that NeuroBridge suffered a major student-data breach. The conversation spreads across X, TikTok, Reddit, Discord, Instagram, school parent forums, App Store reviews, GitHub issues, and small tech blogs. The model must determine what is real, what is unverified, what is fake, what requires escalation, and what the company should do next.

The scenario is intentionally hostile, noisy, contradictory, multi-platform, and time-sensitive.

---

# Primary User Request to the Model

The tester gives the model this instruction:

> Monitor all public discussion about NeuroBridge today. Detect whether a real security incident happened, identify confirmed facts, separate misinformation from credible evidence, track key accounts, preserve memory across batches, identify student-safety and institutional-risk signals, detect coordinated amplification, and produce an executive incident intelligence report with severity, evidence strength, recommended response actions, and unresolved questions.

---

# Company-Side Memory Given to the Model Before Testing

The model receives this internal memory before the first content batch.

## Company

**Name:** NeuroBridge  
**Product:** AI tutoring and academic planning platform  
**Primary users:** high school students, parents, private tutors, school counselors, college-prep programs  
**Regions:** United States, Canada, Singapore, United Kingdom  
**Platforms:** iOS app, Android app, web app  
**Public website:** neurobridge.example  
**Public support email:** support@neurobridge.example  
**Security contact:** security@neurobridge.example

## Known Product Facts

1. NeuroBridge uses AI chat to help students with academic planning, homework review, essay structure, and tutoring workflows.
2. NeuroBridge stores student name, email, grade level, course interests, tutoring history, parent email, and app usage metadata.
3. NeuroBridge does **not** store full Social Security numbers.
4. NeuroBridge does **not** store credit card numbers directly; payment is handled by a third-party payment processor.
5. NeuroBridge does **not** publicly expose private tutoring chats.
6. NeuroBridge supports school roster imports through CSV upload.
7. Some school administrators can upload student rosters.
8. Counselor notes are optional and only available in certain institutional accounts.
9. User profile pictures are optional.
10. NeuroBridge has an internal incident-response policy requiring escalation within 30 minutes when credible breach indicators appear.

## Known Issues Before Today

1. A school administrator accidentally uploaded a duplicate CSV roster last week.
2. Some users received duplicate welcome emails.
3. Some parent accounts are incorrectly linked to inactive student accounts.
4. A small number of users complained that password-reset emails were delayed.
5. The Android app sometimes displays stale notification badges.

## Known Debunked Claim From Yesterday

A viral post yesterday claimed:

> “NeuroBridge stores full Social Security numbers for every student.”

This claim was investigated and publicly debunked. NeuroBridge does not collect or store full Social Security numbers.

## Known Important Actors

### High-Credibility / High-Impact

1. `@K12PrivacyWatch`  
   Education privacy nonprofit. Careful, evidence-based, widely cited by school districts.

2. `@MayaSecOps`  
   Security engineer who often analyzes breach rumors carefully. High technical credibility.

3. `@StudentDataLawyer`  
   Attorney specializing in student privacy and edtech compliance. Influential among parents and school administrators.

4. `@NorthviewPTA`  
   Parent-teacher association account for a real school using NeuroBridge.

5. `@SummitPrepAcademy`  
   Private college-prep chain using NeuroBridge across multiple locations.

6. `@EduAdminForum`  
   Online community for school IT administrators.

### Mixed Reliability

7. `@TechClassroomDaily`  
   Education technology blog. Usually fast but sometimes publishes before full verification.

8. `@AppStoreLeaks`  
   Posts screenshots from apps. Sometimes accurate, sometimes missing context.

### Low Reliability / High Amplification

9. `@BreachAlertNow`  
   Frequently posts alleged breach claims before verification.

10. `@ExposeEdTech`  
    Ideological anti-edtech account. Often makes broad claims with weak sourcing.

11. `@DarkWebIndex`  
    Claims to monitor underground leak markets but has previously amplified false breach claims.

---

# Platforms Included

The model must process signals from:

1. X-style short posts
2. Reddit-style long threads
3. TikTok captions and comments
4. Instagram posts and comment sections
5. Discord school community messages
6. Parent forum posts
7. App Store and Google Play reviews
8. GitHub issue comments
9. Blog snippets
10. Screenshots embedded in posts
11. CSV-like leaked sample text
12. Multi-language posts
13. Deleted-post references
14. Quote posts
15. Reposted screenshots from unknown origins

---

# Test Timeline

The test covers one simulated day.

## Total Simulated Content Volume

The complete hidden dataset contains:

- 9,500 social posts
- 1,200 comments
- 170 app reviews
- 34 parent forum threads
- 11 blog posts
- 7 screenshot clusters
- 4 alleged leaked data samples
- 3 conflicting technical analyses
- 2 official school statements
- 1 official NeuroBridge status update

The model receives only selected batches, but it must infer risk from incomplete, messy evidence.

---

# Required Model Behaviors

The model must perform the following tasks.

## 1. Evidence Classification

For every major claim, the model must assign:

- **Confirmed**
- **Likely**
- **Possible**
- **Unsupported**
- **False / previously debunked**
- **Needs urgent verification**

The model must not use popularity, repost count, emotional intensity, or influencer size as proof.

## 2. Source Weighting

The model must weight evidence according to:

- source credibility
- technical specificity
- firsthand vs. secondhand reporting
- screenshot reliability
- timestamp freshness
- consistency with known internal memory
- whether the post includes verifiable details
- whether the account has a history of accurate or inaccurate claims
- whether the content appears coordinated

## 3. Memory Use

The model must remember:

- yesterday’s debunked Social Security number claim
- known duplicate welcome email issue
- known roster-upload workflow
- known delayed password-reset emails
- important accounts and their credibility levels
- claims made earlier in the day
- whether a claim evolved, weakened, or strengthened over time

## 4. Incident Severity Detection

The model must classify each risk using this scale:

| Level | Meaning |
|---|---|
| SEV-0 | Confirmed active compromise or exposed sensitive student data |
| SEV-1 | Credible evidence of breach, requires urgent incident response |
| SEV-2 | Possible privacy/security issue with incomplete evidence |
| SEV-3 | Reputation issue, misinformation, or product confusion |
| SEV-4 | Low-priority noise or unrelated commentary |

## 5. Stakeholder Risk Identification

The model must separately track risk to:

- students
- parents
- school administrators
- school districts
- private tutoring centers
- legal/compliance teams
- customer support
- engineering/security teams
- app store reputation
- press/media perception
- institutional renewals

## 6. Coordinated Activity Detection

The model must detect:

- copied wording
- synchronized posting
- low-account-age amplification
- repeated use of identical screenshots
- misleading cropped screenshots
- quote-post brigading
- fake “everyone is deleting it” claims
- bot-like repetition
- real user concerns hidden inside noisy amplification

## 7. Information Acquisition Breadth

The model must extract useful signals from:

- short posts
- long-form threads
- app reviews
- comments
- screenshots
- alleged leaked data samples
- technical analysis
- school statements
- parent discussions
- multilingual posts

## 8. Final Report Generation

The final output must include:

1. executive summary
2. confirmed facts
3. unconfirmed but urgent claims
4. false or unsupported claims
5. top risks ranked by severity
6. stakeholder impact map
7. important actors
8. timeline of narrative evolution
9. coordinated activity assessment
10. recommended next actions
11. unresolved questions
12. internal escalation recommendation

---

# Stage 1: Early Rumor Formation

## Input Batch A
## Time Window: 7:00 AM – 9:00 AM

### Post 1 — X

`@BreachAlertNow`:

> BREAKING: NeuroBridge student data may be circulating online. Names, emails, schools, and “academic notes” allegedly exposed. Developing.

### Post 2 — X Reply

> Source?

### Post 3 — X Reply

`@BreachAlertNow`:

> Multiple screenshots. Cannot post all yet.

### Post 4 — Reddit

Title:

> Anyone else get weird duplicate emails from NeuroBridge?

Body:

> My daughter got two welcome emails last week even though we only signed up once. Now I’m seeing people say there was a breach. Are these connected?

### Post 5 — Parent Forum

> Our school uses NeuroBridge for tutoring signups. I saw a post saying student data leaked. Has anyone heard from the school?

### Post 6 — X

`@ExposeEdTech`:

> This is why AI tutoring platforms should not exist. They harvest student data, profile kids, and then act surprised when it leaks.

### Post 7 — App Store Review

> I got password reset emails late last night. I did not request them. Not sure if bug or someone trying to get into my account.

### Post 8 — Discord

> Did anyone else get logged out of NeuroBridge this morning?

### Post 9 — X

`@K12PrivacyWatch`:

> We are aware of claims involving NeuroBridge. At this stage, we have not verified a breach. Parents and schools should preserve screenshots and avoid spreading unconfirmed details.

### Post 10 — Blog Snippet

`TechClassroomDaily`:

> Alleged NeuroBridge breach sparks concern among parents. Evidence remains limited.

## Required Model Output After Stage 1

The model must produce a short incident triage update containing the following classifications.

| Claim | Required Classification |
|---|---|
| Student data is circulating online | Possible / needs urgent verification |
| Duplicate welcome emails prove breach | Unsupported |
| Password-reset emails may indicate account probing | Possible |
| NeuroBridge stores full SSNs | False / previously debunked |
| AI tutoring platforms harvest all student data | Unsupported broad claim |
| Parents are worried | Confirmed |
| Schools need guidance | Likely |

### Required Severity

The model must assign:

> Current severity: **SEV-2**

Reason:

- breach not confirmed
- credible enough to investigate immediately
- student data context increases sensitivity
- password-reset anomaly requires attention
- high parent concern is emerging

### Required Immediate Actions

The model must recommend:

1. open internal incident-response review
2. preserve all public evidence
3. check authentication logs
4. check password-reset volume
5. check recent roster uploads
6. prepare a holding statement
7. contact school administrators privately
8. avoid confirming breach publicly before evidence exists

---

# Stage 2: Alleged Leak Sample Appears

## Input Batch B
## Time Window: 10:00 AM – 12:00 PM

### Post 11 — X Screenshot Post

`@DarkWebIndex`:

> Sample from alleged NeuroBridge leak. Looks real.

Screenshot text:

```csv
student_name,email,school,grade,course_interest,parent_email
Ava M.,ava.m*****@mail.com,Northview High,11,AP Biology,parent.ava*****@mail.com
Leo C.,leo.c*****@mail.com,Northview High,10,SAT Math,parent.leo*****@mail.com
Mina R.,mina.r*****@mail.com,Northview High,12,College Essays,parent.mina*****@mail.com
```

### Post 12 — Reddit Comment

> That looks like a school roster CSV, not necessarily a breach. Admins upload those all the time.

### Post 13 — X

`@MayaSecOps`:

> The alleged NeuroBridge sample is not enough to confirm a platform breach. It could be a roster export, admin mishandling, test data, or fabricated data. Need timestamps, source path, and uniqueness checks.

### Post 14 — Parent Forum

> My son attends Northview High and uses NeuroBridge. His name is not in the sample, but the school name is real.

### Post 15 — Instagram Comment

> They leaked children’s grades and emails. This is insane.

### Post 16 — X

`@AppStoreLeaks`:

> Screenshot shows student names, emails, schools, grade levels, course interests, and parent emails. NeuroBridge needs to answer today.

### Post 17 — Discord

> The CSV fields look exactly like the import template our counselor used.

### Post 18 — X

`@StudentDataLawyer`:

> If the sample is authentic and came from NeuroBridge systems, this may trigger contractual, state, and school notification obligations. Authenticity and source remain unverified.

### Post 19 — TikTok Caption

> NeuroBridge leaked student data?? Parents check your emails.

### Post 20 — App Store Review

> Deleting this app. I don’t care if it’s confirmed or not. Student data should not be in some AI app.

## Required Model Output After Stage 2

The model must update the incident assessment.

### Required Classification

| Claim | Required Classification |
|---|---|
| A CSV-like sample exists online | Confirmed |
| The sample came from NeuroBridge systems | Possible / unverified |
| The sample could be roster-import data | Likely possible explanation |
| The data is highly sensitive | Confirmed, moderate-to-high sensitivity |
| Full tutoring chats leaked | Unsupported |
| Payment data leaked | Unsupported |
| Full SSNs leaked | False / previously debunked unless new evidence appears |
| School-specific exposure risk exists | Likely |

### Required Severity

The model must update severity to:

> Current severity: **SEV-1**

Reason:

- alleged sample contains student-related data
- source is unverified but plausible
- Northview High is a real affected stakeholder
- legal/compliance implications exist
- immediate internal escalation is required

### Required Handling of the CSV Sample

The model must not overstate authenticity.

It must say:

- the sample structure resembles roster data
- the existence of a screenshot does not prove system compromise
- matching field names to known import workflows increases plausibility
- authenticity requires internal log and data-matching verification

### Required Recommended Actions

The model must recommend:

1. compare leaked sample fields to internal roster-import schema
2. check whether the sample rows match real production records
3. identify whether Northview High uploaded a roster recently
4. inspect admin export logs
5. inspect access logs for school admin accounts
6. review object storage and database access logs
7. contact Northview High through official channels
8. prepare regulator/legal notification analysis
9. create public holding statement acknowledging investigation without confirming breach

---

# Stage 3: Coordinated Panic and False Claims

## Input Batch C
## Time Window: 1:00 PM – 3:00 PM

The model receives 2,000 short posts. Selected examples follow.

### Post 21

> NeuroBridge leaked every student’s SSN. Shut it down.

### Post 22

> My school uses NeuroBridge. Are our college essays public now?

### Post 23

> They have parent emails and student grades. This is a disaster.

### Post 24

> Everyone is deleting NeuroBridge today.

### Post 25

> NeuroBridge breach confirmed. Full database leaked.

### Post 26

> The sample only shows 3 students. Could be fake.

### Post 27

> Same CSV screenshot has been reposted 900 times with different captions.

### Post 28

> I saw a version of the screenshot where the school name was changed. Weird.

### Post 29

> My password reset email came 6 hours late. Their email system is broken.

### Post 30

> Why would an AI tutor need parent emails?

### Post 31

> Northview PTA needs to say something.

### Post 32

> The “leak” screenshot has cropped-out rows. I want the original source.

### Post 33

> Students deserve privacy. Even if this is fake, these companies collect too much.

### Post 34

> NeuroBridge stores psychological profiles. That’s what academic notes means.

### Post 35

> My counselor said NeuroBridge only has course preferences and tutoring history for our school.

### Post 36

> Full credit cards leaked too. Saw it on Discord.

### Post 37

> This is exactly like the BrightPath breach last year.

### Post 38

> Why are all these accounts posting the same “full database leaked” phrase?

### Post 39

> NeuroBridge did not leak anything. This is fake outrage.

### Post 40

> Even if it is just a roster, that is still student information.

## Hidden Pattern in Batch C

The system should infer or be told these metadata facts:

1. 1,100 posts use one of five nearly identical phrases:
   - “Full database leaked.”
   - “Every student’s SSN leaked.”
   - “NeuroBridge exposed college essays.”
   - “Credit cards and student files leaked.”
   - “Delete NeuroBridge now.”
2. 620 of those accounts were created within the last 14 days.
3. 480 posts reuse the same CSV screenshot.
4. 90 posts use a modified screenshot where the school name changes from “Northview High” to “North Valley High.”
5. 160 posts ask legitimate parent or student questions.
6. 75 posts mention delayed password-reset emails.
7. 40 posts mention duplicate welcome emails.
8. 25 posts mention possible school admin roster-upload mishandling.
9. 12 posts include technical analysis.
10. 7 posts claim “credit cards leaked” but provide no evidence.

## Required Model Output After Stage 3

The model must produce a structured risk update.

### Required Coordinated Activity Finding

The model must identify:

> Probable coordinated amplification around unsupported claims.

The model must not dismiss all criticism as bots.

It must separate:

| Signal Type | Required Treatment |
|---|---|
| Repeated SSN claims | False / previously debunked |
| Repeated credit-card claims | Unsupported |
| Repeated full-database claims | Unsupported |
| CSV screenshot reposting | Confirmed amplification, authenticity unresolved |
| Modified screenshots | Manipulation risk |
| Parent questions | Legitimate stakeholder concern |
| Delayed password-reset emails | Possible operational/security signal |
| Roster-upload mishandling theory | Plausible investigation path |

### Required Severity

Severity remains:

> **SEV-1**

The model must not escalate to SEV-0 because there is no confirmed active compromise or verified broad database leak.

### Required Harsh Requirements

The model must explicitly state:

1. The social volume is not evidence of breach scope.
2. Repetition is not verification.
3. Modified screenshots lower confidence in some viral claims.
4. Real student-data risk remains possible.
5. The CSV sample must be treated as sensitive until disproven.
6. False claims about SSNs must be corrected using prior memory.
7. Parent and school concern is real regardless of breach confirmation.

---

# Stage 4: Official School Statement and Conflicting Technical Evidence

## Input Batch D
## Time Window: 4:00 PM – 6:00 PM

### Post 41 — Official School Statement

`@NorthviewPTA`:

> We are aware of online claims involving NeuroBridge and Northview High. We have contacted school administration and NeuroBridge. At this time, we have not confirmed unauthorized access. Families will receive an update when more information is available.

### Post 42 — School IT Forum

`@EduAdminForum`:

> Several admins report that NeuroBridge roster templates include exactly these fields: student_name, email, school, grade, course_interest, parent_email. This does not confirm a breach, but the sample format is plausible.

### Post 43 — X

`@MayaSecOps`:

> I found two versions of the “leak sample” screenshot with different school names but identical student initials. That suggests at least some circulating images are altered. Still need to verify whether the original sample is real.

### Post 44 — X

`@K12PrivacyWatch`:

> Current NeuroBridge evidence supports urgent investigation, not public certainty. Avoid sharing student names or screenshots. Schools should ask vendor for log review and data-scope confirmation.

### Post 45 — Blog Snippet

`TechClassroomDaily`:

> NeuroBridge breach may affect thousands, according to viral posts.

### Post 46 — X Reply to Blog

> Your article says “may affect thousands” but only shows the same 3-row screenshot everyone else has.

### Post 47 — Parent Forum

> Our principal emailed families saying they are checking whether the screenshot came from a school roster export.

### Post 48 — Discord

> I’m in a student server. People are sharing the screenshot with full names uncensored. Mods are deleting it.

### Post 49 — App Store Review

> I was worried but the school email says nothing is confirmed. Still, NeuroBridge needs to explain what data schools upload.

### Post 50 — X

`@StudentDataLawyer`:

> If a vendor cannot quickly determine whether a sample belongs to its production data, that itself becomes a governance concern.

## Required Model Output After Stage 4

### Required Updated Assessment

The model must state:

- unauthorized access is still unconfirmed
- the roster-template explanation gained strength
- screenshot manipulation is confirmed for some copies
- the original sample remains unresolved
- school and legal pressure increased
- sharing uncensored student screenshots creates secondary harm
- NeuroBridge must now address data governance, not only breach/no-breach status

### Required Severity

Severity remains:

> **SEV-1**

The model must explain that SEV-1 is maintained because:

- student-related data sample remains plausible
- school stakeholders are involved
- legal/compliance stakeholders are now engaged
- no verified full database leak exists
- no verified active compromise exists

### Required Recommended Actions

The model must recommend:

1. ask users not to share screenshots containing student data
2. issue a public investigation update
3. provide schools with a private admin-specific update
4. verify whether the original sample matches production data
5. audit roster import/export paths
6. audit admin account access
7. reset credentials only if evidence supports compromise
8. avoid mass panic actions without evidence
9. create a parent-facing FAQ
10. preserve evidence for legal/security review

---

# Stage 5: Internal Verification Result Appears

## Input Batch E
## Time Window: 7:00 PM – 8:30 PM

The model receives the following verified internal update.

## Internal Security Update

NeuroBridge security team has completed an initial investigation.

### Verified Findings

1. No evidence of database-wide exfiltration.
2. No evidence that payment data was accessed.
3. No evidence that tutoring chat logs were accessed.
4. No evidence that full Social Security numbers exist in NeuroBridge systems.
5. The original 3-row sample matches a real Northview High roster import from 9 days ago.
6. The roster was uploaded by a Northview school administrator account.
7. That administrator account had a suspicious login from an unfamiliar IP address 2 days ago.
8. Export logs show one CSV export from that administrator account 2 days ago.
9. The export contained 184 student rows.
10. The exported fields were:
    - student name
    - student email
    - school
    - grade
    - course interest
    - parent email
11. No counselor notes were exported.
12. No essay drafts were exported.
13. No tutoring chat messages were exported.
14. No password hashes were exported.
15. Delayed password-reset emails were caused by email queue latency and are unrelated to the admin account issue.
16. Duplicate welcome emails are unrelated to the incident.
17. Some circulating screenshots were altered.
18. The incident appears limited to one compromised school administrator account.
19. Scope may change if further evidence appears.

## Required Model Output After Stage 5

The model must update the incident status with strict precision.

### Required Confirmed Facts

The model must identify as confirmed:

1. A real data exposure occurred.
2. The exposure involved Northview High.
3. The exposure involved one admin account.
4. 184 student roster rows were exported.
5. Exposed fields included student names, student emails, school, grade, course interest, and parent emails.
6. No payment data exposure was found.
7. No tutoring chat exposure was found.
8. No essay draft exposure was found.
9. No counselor note exposure was found.
10. No full SSN exposure exists.
11. Password-reset email delays were unrelated.
12. Duplicate welcome emails were unrelated.
13. Some viral screenshots were altered.

### Required Severity Update

The model must update severity to:

> **SEV-0 for confirmed limited student-data exposure**  
> **Not SEV-0 for full-platform breach**

It must state both parts.

The model must avoid saying:

- “full database breach”
- “all students affected”
- “payment data leaked”
- “tutoring chats leaked”
- “college essays leaked”
- “SSNs leaked”

### Required Incident Label

The model must label the incident as:

> Confirmed limited student roster exposure through likely compromised school administrator account.

### Required Next Actions

The model must recommend:

1. notify Northview High immediately
2. disable or secure the affected admin account
3. rotate session tokens for that admin account
4. review all admin sessions at Northview
5. preserve IP, access, and export logs
6. determine whether notification obligations apply
7. draft parent notification language
8. prepare public statement with precise scope
9. create support workflow for affected families
10. monitor for further reposting of student data
11. request removal of uncensored screenshots where possible
12. continue investigation for lateral movement
13. check all other school admin accounts for similar suspicious logins
14. document unrelated issues separately
15. correct false viral claims without minimizing the real exposure

---

# Stage 6: Final Executive Request

At 9:00 PM, the user asks:

> Give me the final incident intelligence report. I need exact confirmed facts, false claims, current severity, affected stakeholders, key accounts, timeline, evidence strength, coordinated amplification analysis, and tomorrow morning’s operating plan. Be precise. Do not overstate or minimize.

---

# Required Final Report Structure

The final model output must follow this structure exactly.

---

## 1. Executive Summary

Must include:

- confirmed limited student roster exposure
- likely compromised school administrator account
- 184 affected student rows
- Northview High affected
- no evidence of full-platform breach
- no evidence of payment, SSN, tutoring chat, essay draft, or counselor-note exposure
- misinformation amplified the incident beyond verified scope
- parent and school trust impact is severe

---

## 2. Current Severity

Must include:

| Scope | Severity |
|---|---|
| Limited confirmed student roster exposure | SEV-0 |
| Full-platform database breach | Not confirmed |
| Payment data exposure | Not confirmed |
| Tutoring chat exposure | Not confirmed |
| SSN exposure | False / impossible based on known collection |
| Reputation and institutional trust risk | High |

---

## 3. Confirmed Facts

Must list:

1. A real exposure occurred.
2. Original leaked sample matched production roster data.
3. Northview High is affected.
4. The exposure came through a school administrator account.
5. Suspicious login occurred 2 days ago.
6. One CSV export occurred 2 days ago.
7. Export contained 184 student rows.
8. Exposed fields:
   - student name
   - student email
   - school
   - grade
   - course interest
   - parent email
9. No payment data was accessed.
10. No tutoring chats were accessed.
11. No essay drafts were accessed.
12. No counselor notes were exported.
13. No password hashes were exported.
14. No full SSNs exist in the system.
15. Password-reset delays were unrelated.
16. Duplicate welcome emails were unrelated.
17. Some screenshots were altered.

---

## 4. Unsupported, False, or Misleading Claims

Must classify:

| Claim | Classification |
|---|---|
| Full database leaked | Unsupported |
| Every student affected | Unsupported |
| SSNs leaked | False / previously debunked |
| Payment data leaked | Unsupported / contradicted by internal findings |
| Tutoring chats leaked | Unsupported / contradicted by internal findings |
| College essays leaked | Unsupported / contradicted by internal findings |
| Counselor notes leaked | Unsupported / contradicted by internal findings |
| Password reset delays prove breach | False connection |
| Duplicate welcome emails prove breach | False connection |
| No incident happened at all | False |
| All screenshots are authentic | False |
| All screenshots are fake | False |

---

## 5. Evidence Strength Matrix

Must include:

| Finding | Evidence Strength |
|---|---|
| Limited Northview roster exposure | High |
| Admin account compromise | High but still under investigation |
| Full-platform breach | Low |
| Payment data exposure | Low / contradicted |
| Tutoring chat exposure | Low / contradicted |
| SSN exposure | False based on system design |
| Screenshot manipulation | High |
| Coordinated amplification | High |
| Parent concern | High |
| School trust damage | High |
| Password-reset issue unrelated | High |
| Duplicate-email issue unrelated | High |

---

## 6. Timeline of Narrative Evolution

Must include:

### 7:00 AM – 9:00 AM

- breach rumor begins
- password-reset concern appears
- parent concern begins
- `@K12PrivacyWatch` urges caution

### 10:00 AM – 12:00 PM

- CSV-like sample appears
- roster-template theory appears
- legal/compliance concern increases
- `@MayaSecOps` warns evidence is insufficient

### 1:00 PM – 3:00 PM

- coordinated amplification begins
- SSN, credit-card, full-database, essay-leak claims spread
- modified screenshots appear
- legitimate parent questions remain mixed with misinformation

### 4:00 PM – 6:00 PM

- Northview PTA acknowledges concern
- roster-template explanation strengthens
- screenshot manipulation confirmed for some copies
- governance pressure increases

### 7:00 PM – 8:30 PM

- internal investigation confirms limited roster exposure
- 184 rows affected
- likely compromised admin account
- broad viral claims contradicted

---

## 7. Key Actors

Must include:

### `@K12PrivacyWatch`

Role:

- high-credibility privacy nonprofit
- urged caution
- useful for accurate external framing

Required handling:

- provide precise scope
- avoid vague PR language

### `@MayaSecOps`

Role:

- credible technical analyst
- correctly warned against premature conclusions
- identified altered screenshots

Required handling:

- provide technical facts and scope boundaries

### `@StudentDataLawyer`

Role:

- legal/compliance pressure source
- framed notification obligations

Required handling:

- legal team should monitor and prepare precise response

### `@NorthviewPTA`

Role:

- directly affected stakeholder channel
- high parent-trust impact

Required handling:

- highest communication priority

### `@EduAdminForum`

Role:

- school IT/admin audience
- amplified roster-template plausibility

Required handling:

- provide admin-specific security guidance

### `@BreachAlertNow`

Role:

- early rumor amplifier
- partially directionally correct but overbroad

Required handling:

- do not center response around them

### `@DarkWebIndex`

Role:

- shared alleged sample
- source reliability mixed/low

Required handling:

- preserve evidence, avoid treating account as authoritative

### `@ExposeEdTech`

Role:

- ideological amplification
- broad anti-edtech framing

Required handling:

- monitor narrative, do not engage directly unless necessary

### `TechClassroomDaily`

Role:

- fast-moving blog
- published speculative scale language

Required handling:

- provide correction with exact scope

---

## 8. Stakeholder Impact Map

Must include:

| Stakeholder | Impact |
|---|---|
| Affected students | Exposure of roster identity and academic-interest metadata |
| Parents | Parent email exposure and trust loss |
| Northview High | Direct institutional incident |
| Other schools | Concern about whether they are affected |
| School administrators | Need account-security review |
| NeuroBridge security team | Incident containment and evidence preservation |
| NeuroBridge legal/compliance | Notification and contractual analysis |
| Customer support | Parent and school inquiry load |
| Product team | Admin export governance review |
| Executive team | Trust, renewal, and institutional reputation risk |

---

## 9. Coordinated Amplification Analysis

Must include:

- 1,100 posts repeated near-identical claims
- 620 accounts were recently created
- 480 posts reused the same screenshot
- 90 modified screenshots changed the school name
- false claims included SSNs, credit cards, essays, full database, and all-student impact
- amplification increased panic but did not define verified scope
- real incident must not be dismissed because misinformation existed
- misinformation must not be accepted because real exposure existed

---

## 10. Required Public Messaging Position

The model must recommend a message with this posture:

- direct
- precise
- non-defensive
- student-centered
- scope-bounded
- avoids vague phrases like “we take privacy seriously” unless paired with concrete facts
- corrects false claims
- acknowledges real exposure
- explains what data was involved
- explains what data was not involved
- states what actions are underway
- gives affected families a support path

---

## 11. Tomorrow Morning Operating Plan

Must include:

### Security

1. disable or secure affected Northview admin account
2. rotate active sessions
3. force password reset for affected admin account
4. review MFA status
5. review Northview admin access logs
6. review all other institutional admin suspicious logins
7. preserve logs
8. investigate lateral movement
9. validate no additional exports occurred
10. monitor for reposting of exposed data

### Legal / Compliance

1. determine notification obligations
2. review school contract requirements
3. prepare parent notification
4. prepare school administrator briefing
5. document incident timeline
6. preserve evidence chain
7. coordinate screenshot takedown requests where possible

### Communications

1. notify Northview High first
2. notify affected families through approved school channel
3. publish precise public statement
4. issue false-claim correction
5. contact high-credibility external accounts with factual scope
6. provide FAQ for schools and parents
7. avoid debating low-reliability viral accounts

### Product / Engineering

1. audit admin export permissions
2. add export alerting
3. add unusual-login detection for school admins
4. review CSV export logging
5. consider approval workflow for large roster exports
6. improve admin security dashboard
7. separate unrelated email latency issue from incident response
8. separate duplicate welcome email bug from incident response

### Support

1. create parent support script
2. create school admin support script
3. prepare escalation path for affected families
4. tag incoming tickets by school
5. track repeated misinformation themes

---

# Harsh Requirements for the Model

The model must satisfy all of the following.

## Precision Requirements

1. Must not say “NeuroBridge was fully breached.”
2. Must not say “all students were affected.”
3. Must not say “payment data leaked.”
4. Must not say “SSNs leaked.”
5. Must not say “tutoring chats leaked.”
6. Must not say “college essays leaked.”
7. Must not say “counselor notes leaked.”
8. Must not call the incident fake after internal confirmation.
9. Must not dismiss parent concern as misinformation.
10. Must not treat the viral screenshot as fully reliable before internal verification.
11. Must not treat altered screenshots as proof that the original was fake.
12. Must not treat delayed password-reset emails as breach evidence after they are confirmed unrelated.
13. Must not treat duplicate welcome emails as breach evidence after they are confirmed unrelated.
14. Must distinguish “confirmed limited exposure” from “unconfirmed broad breach.”
15. Must update earlier assumptions when stronger evidence appears.

## Memory Requirements

The model must remember:

1. SSN claim was debunked before the test began.
2. Duplicate welcome emails were a known issue.
3. Password-reset delays appeared early and were later confirmed unrelated.
4. CSV sample first appeared as unverified.
5. Roster-template explanation gained strength over time.
6. Some screenshots were altered.
7. Internal investigation confirmed original sample matched production data.
8. Internal investigation confirmed 184 affected rows.
9. Northview High became the core affected institution.
10. Severity changed from SEV-2 to SEV-1 to limited SEV-0.

## Source-Weighting Requirements

The model must prioritize:

1. internal security update over social media speculation
2. official school statements over anonymous comments
3. credible technical analysts over viral accounts
4. firsthand parent/school reports over ideological claims
5. repeated screenshots as amplification, not independent evidence
6. altered screenshots as manipulation risk
7. app reviews as user-impact signals, not security proof
8. blogs as narrative indicators, not authoritative evidence

## Efficiency Requirements

The model must:

1. cluster duplicate claims
2. avoid listing hundreds of repeated posts individually
3. surface the top 5–7 operational risks
4. separate social narrative from technical evidence
5. separate confirmed facts from unresolved questions
6. produce executive-ready language
7. provide action sequencing by team
8. avoid emotional or speculative language
9. avoid long generic privacy lectures
10. produce a decision-grade final report

---

# Failure Conditions

The model fails the test if it does any of the following:

1. Confirms a full-platform breach before evidence supports it.
2. Misses the confirmed limited roster exposure.
3. Fails to identify Northview High as the affected institution.
4. Fails to identify 184 affected student rows.
5. Repeats the false SSN claim as true.
6. Repeats the false payment-data claim as true.
7. Repeats the false tutoring-chat leak as true.
8. Repeats the false essay-leak claim as true.
9. Ignores the compromised admin-account path.
10. Treats duplicate welcome emails as breach evidence after they are confirmed unrelated.
11. Treats delayed password-reset emails as breach evidence after they are confirmed unrelated.
12. Fails to detect coordinated amplification.
13. Dismisses the entire event as misinformation.
14. Fails to update severity over time.
15. Fails to distinguish public narrative risk from technical incident scope.
16. Gives only a sentiment summary instead of an incident intelligence report.
17. Provides vague advice without specific team-level actions.
18. Does not separate confirmed facts from unsupported claims.
19. Does not use the internal security update as the highest-confidence source.
20. Minimizes the real exposure because the broad viral claims were false.

---

# Scoring Rubric

## 1. Accuracy — 30 Points

| Criteria | Points |
|---|---:|
| Correctly identifies confirmed limited roster exposure | 5 |
| Correctly identifies 184 affected rows | 4 |
| Correctly identifies Northview High as affected | 3 |
| Correctly rejects full-platform breach claim | 4 |
| Correctly rejects SSN, payment, essay, chat, counselor-note claims | 5 |
| Correctly separates password-reset latency and duplicate emails from incident | 3 |
| Correctly identifies altered screenshots | 3 |
| Correctly labels admin-account compromise as likely incident path | 3 |

## 2. Memory — 20 Points

| Criteria | Points |
|---|---:|
| Uses prior debunked SSN memory | 4 |
| Tracks duplicate welcome email issue from known context | 3 |
| Tracks password-reset issue across stages | 3 |
| Remembers source credibility profiles | 3 |
| Updates claim status as evidence changes | 4 |
| Updates severity correctly across stages | 3 |

## 3. Evidence Reasoning — 20 Points

| Criteria | Points |
|---|---:|
| Distinguishes screenshots from verified evidence | 4 |
| Weighs internal investigation highest | 4 |
| Distinguishes source credibility levels | 3 |
| Separates volume from proof | 3 |
| Handles conflicting evidence correctly | 3 |
| Identifies unresolved questions | 3 |

## 4. Coordinated Activity Detection — 10 Points

| Criteria | Points |
|---|---:|
| Detects near-identical repeated claims | 2 |
| Uses account-age metadata | 2 |
| Identifies repeated screenshot amplification | 2 |
| Identifies altered screenshots | 2 |
| Separates bot-like noise from real parent concern | 2 |

## 5. Stakeholder and Operational Actionability — 20 Points

| Criteria | Points |
|---|---:|
| Identifies affected stakeholder groups | 4 |
| Provides security actions | 4 |
| Provides legal/compliance actions | 3 |
| Provides communications actions | 3 |
| Provides product/engineering actions | 3 |
| Provides support workflow actions | 2 |
| Prioritizes Northview and affected families | 1 |

**Total:** 100 Points

---

# Final Test Prompt

Use this prompt to run the full test:

> You are monitoring public discussion about NeuroBridge, an AI tutoring platform used by students, parents, tutors, and schools. You have prior memory that NeuroBridge does not collect full Social Security numbers and that yesterday’s SSN leak claim was debunked. You also know that duplicate welcome emails and delayed password-reset emails were known issues before today, but their connection to any breach is unverified unless later evidence confirms it. Analyze the following staged multi-platform content batches. Classify claims by evidence strength, detect misinformation and coordinated amplification, preserve memory across batches, identify stakeholders, update severity over time, and produce a final incident intelligence report. Do not use post volume as proof. Do not overstate. Do not minimize. Distinguish confirmed limited exposure from unsupported full-platform breach claims.

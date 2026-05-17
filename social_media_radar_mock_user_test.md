# Social Media Radar AI Model Mock User Test Scenario

## Scenario Name

**Multi-Platform Crisis Radar for a Sustainability App Launch**

---

## 1. Test Objective

This mock user test is designed to stress-test a **Social Media Radar AI model** across multiple dimensions at the same time:

### 1.1 Performance

The model should handle a high-volume, fast-changing social media environment without slowing down or losing structure.

### 1.2 Accuracy

The model should separate real signals from noise, sarcasm, bot-like activity, duplicated claims, misleading posts, and unsupported accusations.

### 1.3 Memory

The model should remember earlier context, previously identified users, known product issues, prior company clarifications, debunked claims, and evolving narratives.

### 1.4 Information Acquisition Breadth and Efficiency

The model should pull useful signals from multiple platforms, different content formats, screenshots, comments, reposts, quote-posts, and cross-platform references. It should prioritize the most important information instead of treating all posts equally.

---

## 2. Scenario Overview

Your company has launched a new sustainability-focused social media app called **Releaf**.

The app allows users to:

- Post environmental actions
- Share eco-friendly lifestyle content
- Use AI-powered sustainability search
- Interact with a map-based community feed
- Like, comment, and engage with other community members

Three days after launch, the app becomes the subject of a complicated online discussion. Some users praise the app, some criticize it, some spread misinformation, and a few coordinated accounts appear to be trying to damage the app's reputation.

The Social Media Radar AI model must monitor the conversation, classify risks, identify real issues, distinguish unsupported claims from verified problems, and recommend action.

---

## 3. Background Context Given to the Model

Before the test begins, the model receives this company-side memory.

### 3.1 Company Information

**Company:** Releaf  
**Product:** Sustainability-focused social media app  
**Current launch status:** Public beta launched 3 days ago  

### 3.2 Core Product Features

- AI sustainability search
- Map-based posts
- Photo posts
- Comments
- Likes
- User profiles
- Community environmental challenges

### 3.3 Known Real Issues

The company already knows about the following real issues:

1. Some users are confused about location privacy.
2. One bug causes slow image upload on older iPhones.
3. Search sometimes returns generic sustainability advice instead of local-specific answers.
4. The app does not sell user data.
5. The app allows users to choose broad location sharing only; exact location is not shown publicly.

### 3.4 Previously Debunked Fake Claim

The following claim was already debunked yesterday:

> “Releaf secretly tracks exact home addresses and sells them to advertisers.”

The company already posted a clarification yesterday stating that this claim is false.

### 3.5 Known Important Users

| Account | Description |
|---|---|
| `@GreenByteDaily` | Mid-sized environmental tech blogger, usually fair and evidence-based |
| `@EcoWatchdog` | Aggressive critic of greenwashing, influential in sustainability circles |
| `@CampusClimateLab` | High-school and college climate organization |
| `@TechTruthLeaks` | Account with history of posting unverified claims |
| `@MayaBuildsApps` | Indie iOS developer who gives technical reviews |
| `@BayAreaEcoClub` | Local sustainability group with real-world event influence |

---

## 4. Platforms Included in the Test

The radar system must monitor content from the following sources:

1. X / Twitter-style short posts
2. Instagram-style comments and captions
3. TikTok-style video captions and comments
4. Reddit-style discussion threads
5. App Store reviews
6. Discord community messages
7. Blog article snippets
8. Screenshots embedded inside posts
9. Reposts and quote-posts
10. News-style summaries from small online blogs

The full simulation should represent roughly **1,500 to 5,000 pieces of content** over a 24-hour period. However, the model receives selected content batches in stages.

---

## 5. Main User Goal

The user testing the model asks:

> “Monitor everything being said about Releaf today. Identify what actually matters, separate fake claims from real product issues, detect emerging reputation risks, remember previous false claims, find influential voices, summarize the narrative evolution, and recommend what the team should do next.”

This is intentionally broad and difficult. A strong radar model should not merely summarize posts. It should prioritize, classify, remember, and reason.

---

# Stage 1: Morning Signal Burst

## 6. Input Batch A: 8:00 AM to 10:00 AM

### Post 1 — X

> `@TechTruthLeaks`: “So this new Releaf app claims to be ‘eco-friendly’ but asks for location. We’ve seen this story before. Who’s buying the data?”

### Post 2 — X Quote Post

> `@GreenByteDaily`: “I tested Releaf. The app does ask for location permission, but I didn’t see evidence it exposes exact addresses. Still, the privacy wording could be clearer.”

### Post 3 — Reddit

> “Has anyone used Releaf? I like the idea, but the AI search feels kind of shallow. I asked how to reduce plastic use in Palo Alto and it gave me a generic answer.”

### Post 4 — App Store Review

> “Cool concept, but uploading a photo took forever on my iPhone XR. 3 stars until fixed.”

### Post 5 — Instagram Comment

> “Wait does this show where I live?? I want to post but not if strangers can see my neighborhood.”

### Post 6 — Discord

> “The map feature is useful but I think the onboarding screen makes location sound scarier than it is.”

### Post 7 — TikTok Caption

> “Trying Releaf for a week 🌱 first impression: cute app, but not sure if the AI is actually smart yet.”

### Post 8 — Blog Snippet

> “New sustainability app Releaf launches with AI search and community map features. Early users praise its mission but raise questions about privacy clarity.”

---

## 7. Expected Model Behavior in Stage 1

The model should identify three important early themes.

### 7.1 Theme 1: Privacy Confusion

This is not necessarily a confirmed privacy violation. It is a **communication and trust issue**.

Correct classification:

> Medium reputational risk, low evidence of actual privacy abuse.

### 7.2 Theme 2: AI Search Quality Weakness

This issue is real and product-related.

Correct classification:

> Confirmed product quality issue, moderate user experience impact.

### 7.3 Theme 3: Image Upload Slowness

This issue is already known and appears again.

Correct classification:

> Known technical bug recurring in user feedback.

### 7.4 Memory Expectation

The model should remember that:

- `@TechTruthLeaks` is not highly reliable.
- The exact-address claim was previously debunked.
- Privacy confusion is real, but data-selling or exact-address exposure is not verified.

---

# Stage 2: Midday Escalation

## 8. Input Batch B: 12:00 PM to 2:00 PM

### Post 9 — X

> `@TechTruthLeaks`: “Releaf quietly updated its privacy page after people called them out. That usually means they were hiding something.”

### Post 10 — Screenshot Post

A user posts a screenshot of an old onboarding screen:

> “Allow location so Releaf can connect you with nearby eco-actions.”

Caption:

> “This sounds like exact tracking to me.”

### Post 11 — Reddit Comment

> “I checked their privacy policy. It says broad location, not exact GPS. But the app permission popup on iPhone still says location, which scares people.”

### Post 12 — Instagram

> `@BayAreaEcoClub`: “We want to use Releaf for a cleanup event this weekend, but members are asking how public location posts work. Can someone explain?”

### Post 13 — TikTok Comment

> “They definitely sell your house location. My cousin said all these apps do.”

### Post 14 — X

> `@MayaBuildsApps`: “Technical note: iOS location permission wording often sounds harsher than what the app actually displays publicly. Releaf should show a plain-English privacy explanation before requesting permission.”

### Post 15 — Discord

> “Image upload is still slow. Is it compressing locally or sending full-res first?”

### Post 16 — App Store Review

> “The app is promising, but the AI gave the same advice for NYC and San Jose. Needs local data.”

---

## 9. Expected Model Behavior in Stage 2

The model should detect that the privacy narrative is escalating but still lacks strong evidence of actual wrongdoing.

It should distinguish between the following claims:

### Claim A

> Users are confused and worried about location.

This is true.

### Claim B

> Releaf sells exact home addresses.

This is unsupported and was previously debunked.

### 9.1 Important Actor Detection

The model should identify `@BayAreaEcoClub` as operationally important because they may influence real-world adoption, even if they are not the largest account.

The model should also notice that `@MayaBuildsApps` provides a high-quality technical interpretation and should be weighted more heavily than casual comments.

---

# Stage 3: Coordinated Noise and Bot-Like Activity

## 10. Input Batch C: 3:00 PM to 5:00 PM

The system receives 500 short posts. Many look similar.

Examples:

### Post 17

> “Releaf = location stealing app. Delete now.”

### Post 18

> “Another fake green app stealing your data.”

### Post 19

> “Releaf wants your home address. Wake up.”

### Post 20

> “Eco app? More like tracking app.”

### Post 21

> “Funny how they say broad location but ask for location permission.”

### Post 22

> “I don’t trust any app that needs location.”

### Post 23

> “Releaf is probably selling data. No proof but obvious.”

### Post 24

> “Stop using Releaf until they explain location.”

### Post 25

> “I asked their AI about recycling near me and it was wrong lol.”

### Post 26

> “Upload bug is real. Took 2 minutes to post a photo.”

---

## 11. Hidden Pattern in Batch C

About 300 of the 500 posts use highly similar wording:

> “Releaf wants your home address.”  
> “Releaf steals your address.”  
> “Releaf tracks your home.”  
> “Releaf sells your location.”

Many accounts were created within the last 7 days, have low follower counts, and repost each other.

However:

- About 80 posts are legitimate users asking reasonable privacy questions.
- About 50 posts mention real product bugs.
- About 20 posts are sarcastic and should not be classified literally.

Example sarcastic post:

> “Yeah because obviously my compost photo is a CIA operation.”

---

## 12. Expected Model Behavior in Stage 3

This is the stress-test phase.

The model should:

1. Detect probable coordinated amplification.
2. Avoid dismissing all criticism as bots.
3. Separate bot-like misinformation from genuine user concern.
4. Detect sarcasm.
5. Preserve the real issues:
   - Unclear privacy communication
   - Generic AI search
   - Slow uploads
6. Avoid overreacting to unsupported claims.
7. Escalate the reputation risk because the false narrative is spreading.

Correct classification:

> High narrative risk, medium product trust risk, low evidence of confirmed data misuse.

The model should recommend that the team respond quickly with a clear privacy explainer, not with a defensive corporate statement.

---

# Stage 4: Conflicting Evidence Appears

## 13. Input Batch D: 6:00 PM to 8:00 PM

### Post 27 — Blog Article

> “Users accuse Releaf of location tracking, but available evidence suggests the controversy may be driven by unclear onboarding wording rather than actual misuse.”

### Post 28 — X

> `@EcoWatchdog`: “I’m not saying Releaf sells data. I am saying sustainability apps must earn trust. Their location explanation is too vague.”

### Post 29 — X Reply

> “Finally someone said it. Even if they aren’t selling data, why should an eco app need a map?”

### Post 30 — Reddit Technical Analysis

> “I inspected network traffic briefly. I didn’t see exact GPS coordinates being sent in the public posting flow, but I only tested one device and one session.”

### Post 31 — Discord

> “The devs said location is optional, but I skipped it and then some map features were empty. That makes sense but should be explained better.”

### Post 32 — App Store Review

> “I changed my rating from 2 to 4 after reading the privacy explanation. Still want better AI search though.”

### Post 33 — Instagram

> `@CampusClimateLab`: “We are pausing our Releaf campaign until privacy questions are clearly answered. We like the mission but need clarity for student users.”

### Post 34 — TikTok

> “I tested Releaf and honestly the privacy panic seems exaggerated. The AI search is the weaker part.”

---

## 14. Expected Model Behavior in Stage 4

The model should update its understanding.

Privacy remains important, but the narrative is becoming more nuanced.

The strongest confirmed issue is now not privacy abuse, but:

> Insufficiently clear location-permission explanation.

The model should also identify a major stakeholder risk:

> `@CampusClimateLab` pausing its campaign is important because it affects organized adoption among student climate communities.

The model should remember that `@EcoWatchdog` is influential and did not accuse Releaf of selling data. Misquoting this account would be a serious accuracy failure.

---

# Stage 5: Final Executive Request

## 15. Final User Request at 9:00 PM

At 9:00 PM, the user asks the model:

> “Give me a final crisis intelligence report. I need the top risks, what is real, what is fake, who matters most, what changed throughout the day, and what we should do tomorrow morning.”

The model must generate a structured executive-level report.

---

# 16. Expected Final Output From the Model

A strong model should produce a report similar to the following.

## 16.1 Executive Summary

Releaf experienced a fast-moving reputation incident centered on location privacy. The dominant online narrative was not caused by confirmed privacy misuse, but by vague onboarding language, iOS permission anxiety, and coordinated amplification of an already-debunked claim that Releaf sells exact home addresses.

The most credible product issues are generic AI search quality and slow image upload on older devices.

---

## 16.2 Top Risks

### Risk 1: Location Privacy Trust Risk

**Severity:** High  
**Evidence strength:** Medium for user confusion, low for actual misuse  
**Core issue:** Users do not understand what location data is collected, how precise it is, and what is publicly visible.

### Risk 2: Coordinated Misinformation Risk

**Severity:** High  
**Evidence strength:** Medium-high  
**Core issue:** Many similar posts repeat claims about home-address tracking without evidence.

### Risk 3: AI Search Quality Risk

**Severity:** Medium  
**Evidence strength:** High  
**Core issue:** Users report generic or non-local answers.

### Risk 4: Image Upload Performance Risk

**Severity:** Medium  
**Evidence strength:** High  
**Core issue:** Older iPhones experience slow upload times.

### Risk 5: Community Adoption Risk

**Severity:** High  
**Evidence strength:** High  
**Core issue:** `@CampusClimateLab` and `@BayAreaEcoClub` may delay adoption unless privacy messaging improves.

---

## 16.3 What Is Real vs. Fake

### Real Issues

1. Users are confused by location permission wording.
2. Releaf’s onboarding does not explain location clearly enough.
3. AI search lacks strong local specificity.
4. Image uploads are slow for some users.
5. Some community groups are delaying adoption because of trust concerns.

### Unsupported or False Claims

1. Releaf sells exact home addresses.
2. Releaf publicly reveals where users live.
3. Releaf secretly changed its privacy policy to hide misconduct.
4. Releaf is confirmed to be a tracking app.

The model should explicitly connect these unsupported claims to the memory that the exact-address claim was already debunked yesterday.

---

## 16.4 Important Actors

### Highest Priority Actors

#### `@CampusClimateLab`

Important because they represent student climate communities and paused a campaign. This is an adoption risk, not just a public relations issue.

#### `@BayAreaEcoClub`

Important because they may use Releaf for an actual cleanup event. Their questions should be answered quickly.

#### `@EcoWatchdog`

Important because they are influential and critical but currently nuanced. They criticized vague wording, not confirmed data abuse.

#### `@GreenByteDaily`

Important because they are balanced and evidence-based. They can help correct misinformation if given clear facts.

#### `@MayaBuildsApps`

Important because they provided technically accurate framing around iOS permission language and user experience design.

### Lower Reliability Actor

#### `@TechTruthLeaks`

Important for narrative spread, but lower reliability because they repeated insinuations without evidence and have a history of unverified claims.

---

## 16.5 Narrative Evolution

### Morning

The conversation began as mixed launch feedback. Users liked the concept but questioned privacy wording, AI quality, and image upload speed.

### Midday

Privacy concerns escalated after screenshots of onboarding language circulated. Technical voices clarified that the issue was likely communication, not proven misuse.

### Afternoon

Bot-like amplification pushed stronger unsupported claims about home-address tracking and data selling. The model should classify this as coordinated narrative risk, not verified evidence.

### Evening

The conversation became more nuanced. Credible accounts criticized unclear privacy language, while some users began correcting misinformation. However, organized community groups paused adoption pending clarification.

---

## 16.6 Recommended Actions for Tomorrow Morning

### Action 1: Publish a Plain-English Location Privacy Explainer

The explainer should answer:

- Is location required?
- What precision is collected?
- What is publicly visible?
- Is exact location shown?
- Is user data sold?
- Can users use Releaf without location?
- Why does the map feature need approximate location?

### Action 2: Update Onboarding Copy

Replace vague language like:

> “Allow location so Releaf can connect you with nearby eco-actions.”

With clearer language like:

> “Releaf uses optional approximate location to show nearby sustainability posts and events. Your exact home address is not shown publicly.”

### Action 3: Reply Directly to Credible Community Accounts

Priority response order:

1. `@CampusClimateLab`
2. `@BayAreaEcoClub`
3. `@GreenByteDaily`
4. `@EcoWatchdog`
5. `@MayaBuildsApps`

### Action 4: Avoid Arguing With Low-Reliability Viral Accounts

Do not center the company response around `@TechTruthLeaks`. Correct the false claim publicly, but avoid boosting them unnecessarily.

### Action 5: Ship Product Fixes

Immediate engineering priorities:

1. Investigate slow image upload on older iPhones.
2. Add image compression before upload if missing.
3. Improve local specificity in AI search.
4. Add a privacy explainer screen before the iOS permission request.
5. Add an in-app FAQ link from the map and posting flow.

### Action 6: Create a Misinformation Response Protocol

The system should flag repeated claims such as:

> “Releaf sells your address.”

as previously debunked misinformation unless new evidence appears.

---

# 17. Evaluation Rubric

Use this rubric to score the model after running the scenario.

## 17.1 Accuracy — 30 Points

The model should correctly separate verified facts from unsupported claims.

| Criteria | Points |
|---|---:|
| Identifies real privacy confusion | 5 |
| Does not falsely confirm data-selling claim | 5 |
| Recognizes AI search weakness as real | 5 |
| Recognizes image upload bug as real | 5 |
| Correctly interprets nuanced critics like `@EcoWatchdog` | 5 |
| Detects sarcasm and does not classify it literally | 5 |
| **Total** | **30** |

---

## 17.2 Memory — 20 Points

The model should use previous context instead of treating every post as new.

| Criteria | Points |
|---|---:|
| Remembers exact-address claim was debunked yesterday | 5 |
| Remembers known product bugs | 5 |
| Tracks important users across stages | 5 |
| Updates risk level as new evidence appears | 5 |
| **Total** | **20** |

---

## 17.3 Information Acquisition Breadth — 20 Points

The model should synthesize across multiple content sources.

| Criteria | Points |
|---|---:|
| Uses X, Reddit, TikTok, Instagram, Discord, App Store, and blog signals | 5 |
| Weighs source credibility differently | 5 |
| Extracts information from screenshots, captions, and comments | 5 |
| Identifies community adoption risk, not just social sentiment | 5 |
| **Total** | **20** |

---

## 17.4 Efficiency — 15 Points

The model should prioritize instead of drowning in data.

| Criteria | Points |
|---|---:|
| Clusters repetitive posts correctly | 4 |
| Detects coordinated amplification | 4 |
| Prioritizes high-impact accounts | 4 |
| Produces concise executive summary | 3 |
| **Total** | **15** |

---

## 17.5 Actionability — 15 Points

The model should recommend practical next steps.

| Criteria | Points |
|---|---:|
| Recommends privacy explainer | 3 |
| Recommends onboarding copy changes | 3 |
| Recommends targeted stakeholder replies | 3 |
| Recommends engineering fixes | 3 |
| Recommends misinformation protocol | 3 |
| **Total** | **15** |

---

## 17.6 Total Score

| Category | Points |
|---|---:|
| Accuracy | 30 |
| Memory | 20 |
| Information Acquisition Breadth | 20 |
| Efficiency | 15 |
| Actionability | 15 |
| **Total** | **100** |

---

# 18. Hard Mode Add-On

To make this test significantly harder, add these complications.

## 18.1 Contradictory Screenshots

One user posts an outdated onboarding screenshot. Another user posts the updated screen.

The model must determine which version is current and avoid treating old screenshots as current product evidence.

## 18.2 Multi-Language Posts

Add comments in Chinese, Spanish, and French asking whether location is public.

Examples:

> “这个软件会不会公开我的位置？”  
> “¿La app muestra mi ubicación exacta?”  
> “Est-ce que l’application montre ma position exacte?”

The model should recognize that all three posts are asking the same privacy question.

## 18.3 Influencer Misquotation

A viral account claims:

> “Even EcoWatchdog said Releaf sells data.”

But `@EcoWatchdog` only said the privacy wording was vague.

The model must catch the misquotation.

## 18.4 Old Bug Resurfacing

Some users complain about a bug that was fixed in version 1.0.2, but they are using version 1.0.0.

The model must separate current-version issues from outdated-version issues.

## 18.5 False Consensus Attack

Hundreds of posts say:

> “Everyone is deleting Releaf.”

But App Store reviews and Discord messages show many users are still using it.

The model should avoid accepting volume as truth.

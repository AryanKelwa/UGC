prompt="""You are a synthetic training data generator for a real estate lead qualification AI fine-tuned for Anywhere Real Estate (brands: Coldwell Banker, Century 21, ERA, Sotheby's International Realty, Better Homes and Gardens Real Estate, Corcoran).

---

OUTPUT FORMAT

Generate exactly 30 fine-tuning examples in JSONL format — one JSON object per line, no extra text before or after.

Each example has exactly two fields:

{
  "user": "Lead conversation:\\n[full conversation]",
  "assistant": "{escaped JSON string}"
}

Do NOT include a "system" field in any example. The system prompt is injected separately at inference time and must not be repeated in training data.

The assistant field must be a single-line escaped JSON string with these keys:
intent, urgency, budget_range, timeline_months, preferred_locality, property_type, brand_fit, next_action, score, reasoning

---

ASSISTANT OUTPUT SCHEMA

intent         → buy / rent / invest / browse
urgency        → hot / warm / cold
budget_range   → stated or inferred string e.g. "$500K–$650K" or "unknown"
timeline_months → integer — estimated months to transaction; null for drop leads
preferred_locality → most specific location available (neighborhood > city > metro)
property_type  → descriptive e.g. "3BR single-family", "luxury penthouse", "12-unit multifamily"
brand_fit      → one of the 6 Anywhere brands, or "N/A" for drop leads
next_action    → agent_call / schedule_showing / follow_up_email / nurture_sequence / drop
score          → integer 0–100 (conversion probability)
reasoning      → 2–4 sentences: name the signals, give a routing instruction

---

CONVERSATION REALISM — THIS IS THE MOST IMPORTANT SECTION

The conversations must read like real humans talking, not like a demo script. Follow every rule below.

NATURAL LANGUAGE:
- People abbreviate, hedge, trail off, and change their minds mid-conversation
- Use contractions always ("I'm", "we're", "don't", "it's")
- Buyers can be vague, imprecise, or emotionally loaded ("we've been looking forever", "I'm so stressed about this")
- Agents should sound warm and professional but not robotic — occasional filler is fine ("Absolutely!", "Got it, that helps a lot")
- Vary sentence length — short replies mix with longer explanatory ones
- Spelling mistakes and informal punctuation are acceptable occasionally

CONVERSATION LENGTH — VARY THIS DELIBERATELY:
- 30% of examples should be SHORT: 3–4 turns. A lead fires off a quick question, agent asks one qualifying question, lead reveals a key signal. Realistic for portal inquiries and WhatsApp pings.
- 50% of examples should be MEDIUM: 5–8 turns. A natural back-and-forth where intent, budget, and timeline emerge gradually.
- 20% of examples should be LONG: 9–14 turns. A full qualification conversation — lead asks multiple questions, compares properties, reveals personal context, maybe pushes back or changes their stated preference mid-thread. These are the richest training examples.

WHAT REAL CONVERSATIONS INCLUDE THAT SCRIPTED ONES DON'T:
- The lead asks about the property first before the agent asks qualifying questions
- The lead sometimes dodges a question or answers a different question than was asked
- The lead sometimes volunteers information the agent didn't ask for ("by the way, we have two dogs")
- The lead sometimes contradicts themselves ("we're not in a rush" → two messages later → "our lease is actually up in 6 weeks")
- The agent sometimes has to re-ask a question because the lead gave a vague answer
- Leads express emotion: excitement, anxiety, frustration, uncertainty
- Some leads have done their homework (mention specific comps, cap rates, DOM); others know almost nothing
- In long conversations, the topic sometimes drifts — a buy lead might ask about the rental market, an investor might ask about property taxes mid-thread
- Conversations can start from different entry points: a specific listing inquiry, a general area question, a form submission reference, a WhatsApp message, a portal callback request

THINGS TO ACTIVELY AVOID:
- Every message being the same length
- The agent always asking questions in the exact same order
- The lead answering every question completely and neatly
- Conversations that feel like a form being filled out verbally
- Urgency signals appearing in the first message every time
- All conversations starting with "Hi, I saw your listing"

---

STRICT DIVERSITY RULES

INTENT MIX (exactly):
- 10 buy leads
- 6 rent leads
- 6 invest leads
- 5 browse leads
- 3 drop leads (journalist, competitor agent, spam, wrong number, already bought elsewhere, etc.)

URGENCY MIX (across buy + rent + invest leads only):
- 7 hot (timeline < 60 days, strong financial signals present)
- 8 warm (2–6 months, moderate signals)
- 7 cold (6+ months, vague or aspirational)

SCORE DISTRIBUTION (track actively — no tier should be empty):
- 4 examples: 85–100 (pre-approved, hard deadline, clear budget, ready now)
- 5 examples: 65–84 (real intent, one friction point)
- 6 examples: 40–64 (genuine but gated — financing pending, contingency, life event)
- 6 examples: 15–39 (early stage, no pre-approval, long horizon)
- 5 examples: 5–14 (browsing, researching, no transaction intent this year)
- 4 examples: 0–4 (true non-leads — drop immediately)

BRAND FIT — max 6 per brand per batch:
Coldwell Banker, Century 21, ERA, Sotheby's International Realty,
Better Homes and Gardens Real Estate, Corcoran
Drop leads → "N/A"

GEOGRAPHY — different US city or metro for every example, no repeats within a batch.
Cover all regions: Northeast, Southeast, Midwest, Southwest, West Coast, Mountain West,
Mid-Atlantic, Pacific Northwest, Gulf Coast, Great Plains.

PROPERTY TYPES — use each at least once per batch:
single-family, condo, townhome, luxury penthouse, small multifamily (2–4 units),
large multifamily (5–20 units), farm/acreage, commercial office or retail,
new construction, co-op, furnished short-term rental, vacation/second home

LEAD PERSONAS — max 2 uses per persona per batch:
corporate relocator, military PCS buyer, first-time buyer, luxury repeat buyer,
foreign national buyer, post-divorce buyer, empty nester downsizer,
remote worker lifestyle mover, new grad renter, travel nurse / contract worker,
fix-and-flip investor, buy-and-hold investor, 1031 exchange buyer,
UHNW second-home buyer, rate-watching hesitant buyer, simultaneous buy-sell,
new construction buyer, commercial tenant, institutional investor (research only),
competitor agent, journalist or market researcher, wrong number / spam

---

SIGNAL RULES BY SCORE TIER

Score > 70 — must include at least one of:
lease ending soon, pre-approved, job start date confirmed, school enrollment deadline,
PCS orders received, baby due date, divorce settlement closing, bridge loan ready,
all-cash buyer, visa or immigration deadline, property visit already scheduled

Score 40–69 — must include at least one of:
contingent on current home sale, no pre-approval yet, rate uncertainty causing hesitation,
down payment funds not yet liquid, spouse or partner not yet aligned, timeline genuinely vague,
credit score borderline, waiting on a life event (job offer, inheritance, divorce settlement)

Score < 40 — lead must clearly show one or more of:
explicitly states long horizon (12+ months or "a few years"), refuses to give financial details,
is researching rather than transacting, has no urgency signal whatsoever,
is gathering information for someone else, already working with another agent

Score 0–4 — make the non-lead status unambiguous within the conversation itself.

---

REASONING FIELD RULES
- 2–4 sentences, no more
- Name the specific signals that drove the score (quote the lead's words if useful)
- Give a concrete routing instruction: which agent type, what to do first, any risk to flag
- For drop leads: why this is not a pipeline lead and whether to log the contact for any other purpose

---

SELF-CHECK BEFORE OUTPUTTING

Before writing your final output, verify:
[ ] Exactly 30 examples, no system field in any of them
[ ] Intent counts: 10 buy, 6 rent, 6 invest, 5 browse, 3 drop
[ ] No city repeated
[ ] No brand used more than 6 times
[ ] Score distribution covers all 6 tiers
[ ] Conversation lengths are mixed — not all the same length
[ ] At least 6 conversations are 9+ turns (the long ones)
[ ] At least one conversation has the lead contradicting themselves
[ ] At least one conversation has the lead asking about the property before being qualified
[ ] Drop leads: score ≤ 4, next_action = "drop", brand_fit = "N/A"
[ ] All assistant fields are valid escaped JSON strings

OUTPUT RULES:
- Pure JSONL only — one JSON object per line
- No markdown, no backticks, no commentary, no blank lines between entries
- No line breaks inside a single JSON object
- All internal quotes in the assistant string must be escaped as \\"

Generate batch number: [1-34]"""
# Exa AI General Usage Guide

This is a reusable guide for using Exa AI in research, search, crawling, enrichment, and directory-building workflows. It is intentionally general; use it alongside project-specific notes such as `MINISTRY_DIRECTORY_PROCESS.md`.

Note: the user may refer to this as "Excel AI" in speech, but the product used in this project is Exa AI.

Official docs checked during this project:

- Exa quickstart: https://exa.ai/docs/reference/quickstart
- Python SDK: https://exa.ai/docs/sdks/python-sdk
- Search API guide: https://exa.ai/docs/reference/search-api-guide-for-coding-agents
- Contents API guide: https://exa.ai/docs/reference/contents-api-guide
- Agent API guide: https://exa.ai/docs/reference/agent-api-guide
- Pricing: https://exa.ai/pricing

## Mental Model

Exa has several related pieces. They use the same Exa API key, but they are useful at different points in a workflow.

| Piece | What it does | Use it when |
| --- | --- | --- |
| API | The raw HTTP endpoints such as `/search`, `/contents`, and `/agent/runs` | You want exact control, no SDK dependency, or a language without a mature SDK |
| SDK | Language wrapper such as `exa-py` or `exa-js` | You want simpler Python or JavaScript code |
| Search | Finds web pages from a natural-language query | You need candidate links, current web results, source discovery, or lightweight grounding |
| Contents | Extracts clean text, highlights, summaries, links, images, and subpages from known URLs | You already have URLs and need to read them |
| Deep Search | A Search `type` that does more synthesis/reasoning than ordinary search | You need structured output or a more complex comparison, but not a long async Agent run |
| Agent | Async research/list-building/enrichment run | You need multi-step research, entity enrichment, list building, schema-validated JSON, citations, or follow-up runs |
| Monitors | Scheduled searches | You need ongoing alerts or recurring web monitoring |
| Websets | Visual/API workflow for finding, verifying, and enriching sets of entities | You want verified, enriched result sets with less custom pipeline code |

For most custom pipelines, the practical default is:

```text
Search -> Contents -> Agent -> audit/review -> export/save
```

## API Key Setup

Use one Exa API key for Search, Contents, Agent, and the SDK.

Store it in `.env`:

```bash
EXA_API_KEY=your_key_here
```

Do not commit or print the key.

Python setup:

```bash
pip install exa-py python-dotenv
```

Minimal Python client:

```python
import os
from dotenv import load_dotenv
from exa_py import Exa

load_dotenv()

api_key = os.getenv("EXA_API_KEY")
if not api_key:
    raise RuntimeError("Missing EXA_API_KEY")

exa = Exa(api_key=api_key)
```

JavaScript setup:

```bash
npm install exa-js dotenv
```

## When To Use Search

Use Search when the starting point is a query rather than a known URL.

Good for:

- Finding candidate websites
- Getting recent news or source pages
- Building a first-pass result list
- Retrieving token-efficient highlights from search results
- Finding official pages for organizations, products, people, papers, events, or documents

Basic Python example:

```python
from exa_py import Exa

exa = Exa(api_key="YOUR_EXA_API_KEY")

result = exa.search(
    "AI infrastructure companies hiring founding designers",
    type="auto",
    num_results=10,
    contents={"highlights": True},
)

for item in result.results:
    print(item.title, item.url)
```

Useful Search options:

```python
result = exa.search(
    "recent AI regulation policy updates",
    type="auto",
    num_results=10,
    include_domains=["reuters.com", "bbc.com"],
    start_published_date="2025-01-01",
    contents={
        "highlights": True,
    },
)
```

For Python SDK calls, prefer snake_case parameter names:

```python
num_results
include_domains
exclude_domains
start_published_date
max_age_hours
max_characters
subpage_target
```

For raw JSON/cURL and JavaScript, use camelCase:

```json
{
  "numResults": 10,
  "includeDomains": ["example.com"],
  "contents": {
    "maxAgeHours": 24
  }
}
```

## Search Types

Start with `type="auto"` unless there is a clear reason not to.

Common types:

- `auto`: good default balance of quality and latency
- `fast`: lower latency
- `instant`: lowest latency for real-time apps
- `deep-lite`: lighter synthesized/deep search
- `deep`: multi-step search with reasoning and structured output
- `deep-reasoning`: maximum reasoning depth for harder tasks

Use `deep` or `deep-reasoning` when you want Search itself to synthesize an answer or populate a schema from multiple sources. Use Agent when the task is broader, async, iterative, or row-enrichment-heavy.

## When To Use Contents

Use Contents when you already know the URL or URLs.

Good for:

- Reading full page text
- Getting targeted highlights
- Summarizing a known page
- Crawling relevant subpages from a site
- Extracting links or image URLs from a page

Important difference:

- On Search, content options are nested under `contents`.
- On Contents, the same options are top-level arguments.

Search example:

```python
result = exa.search(
    "best open source vector databases",
    contents={
        "highlights": True,
        "text": {"max_characters": 5000},
    },
)
```

Contents example:

```python
result = exa.get_contents(
    ["https://example.com/article"],
    highlights={"query": "pricing, features, limitations"},
    text={"max_characters": 5000},
)
```

Subpage crawling example:

```python
result = exa.get_contents(
    ["https://example.com"],
    subpages=10,
    subpage_target=["about", "pricing", "contact", "docs"],
    text={"max_characters": 5000},
    max_age_hours=24,
)
```

Use `highlights` for agent workflows when possible. It is usually much cheaper in tokens than full `text`.

Use full `text` when:

- You do not know which part of the page matters
- You are doing deep analysis
- Highlights are too thin
- The page is short enough to fit comfortably

Use `max_age_hours=0` only when freshness matters. It forces a live crawl and may increase latency. For many workflows, omitting `max_age_hours` or using `24` is enough.

## Can Exa Go Deeper Into Pages?

Yes, but choose the mechanism deliberately.

For deterministic deeper crawling, use Search or Contents with:

```python
contents={
    "highlights": True,
    "subpages": 5,
    "subpage_target": ["about", "team", "pricing", "contact"],
    "extras": {
        "links": 10,
        "image_links": 10,
    },
}
```

Or with known URLs:

```python
exa.get_contents(
    ["https://example.com"],
    subpages=5,
    subpage_target=["about", "team", "pricing", "contact"],
    highlights=True,
)
```

Agent can also perform deeper web research, but it is not a browser you control click-by-click. If exact traversal matters, first use Search/Contents to collect pages, subpages, and links, then give the compact evidence to Agent.

## When To Use Agent

Use Agent when a workflow needs more than one search/extraction call or needs substantial reasoning.

Good for:

- Building a list from open-ended criteria
- Enriching rows you already have
- Comparing entities across many fields
- Producing schema-validated JSON
- Getting citations/grounding for structured fields
- Running a follow-up from a previous run
- Auditing a previous pass

Basic Agent example:

```python
import json
from exa_py import Exa

exa = Exa(api_key="YOUR_EXA_API_KEY")

run = exa.agent.runs.create(
    query="Find AI infrastructure companies hiring founding designers.",
    effort="medium",
    output_schema={
        "type": "object",
        "properties": {
            "companies": {
                "type": "array",
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "website": {"type": "string", "format": "uri"},
                        "reason": {"type": "string"},
                        "source_urls": {
                            "type": "array",
                            "items": {"type": "string", "format": "uri"},
                        },
                    },
                    "required": ["name", "website", "reason", "source_urls"],
                },
            }
        },
        "required": ["companies"],
    },
)

completed = exa.agent.runs.poll_until_finished(run.id)
data = completed.output.structured if completed.output else None
print(json.dumps(data, indent=2))
```

Use `input.data` when enriching rows:

```python
rows = [
    {"id": "1", "name": "Example Co", "website": "https://example.com"},
    {"id": "2", "name": "Another Co", "website": "https://another.example"},
]

run = exa.agent.runs.create(
    query=(
        "For each input row, find the public pricing page and summarize the "
        "main pricing model. Only return rows from the input data."
    ),
    input={"data": rows},
    effort="medium",
    output_schema={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "maxItems": len(rows),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "pricing_url": {"type": "string"},
                        "summary": {"type": "string"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                    },
                    "required": ["id", "name", "pricing_url", "summary", "confidence"],
                },
            }
        },
        "required": ["items"],
    },
)
```

Use `input.exclusion` or your own prompt/exclusion file to avoid re-finding known results.

Use `previous_run_id` for follow-up runs when you want to continue from a previous Agent result.

## Agent Prompting Rules That Matter

Agent is capable and helpful, which means it may also "helpfully" add familiar entities that were not in your candidate set. When curating a known list, be explicit:

```text
Only select organizations that appear in the candidate dossiers below.
You may search to verify or reject dossier organizations, but do not add
familiar organizations, nearby alternatives, or examples that are not
present in the dossiers.
```

For reliable structured outputs:

- Provide an `output_schema`.
- Use `required` fields for anything your app depends on.
- Use enums for statuses, confidence, categories, and review decisions.
- Use `maxItems` to control scope and cost.
- Ask for source URLs or citations for claims.
- Ask for confidence and review notes.
- Tell Agent what to do when evidence is missing.
- Separate "found", "verified", and "safe to publish" as different fields.

Good generic review fields:

```text
name
website
summary
category
source_urls
confidence
review_status
review_notes
last_verified_at
```

Useful review statuses:

```text
publish_ready
manual_review
duplicate
probably_exclude
out_of_scope
insufficient_evidence
```

## Recommended General Workflow

1. Define the target clearly.
2. Define exclusions and non-goals.
3. Run broad Search queries with `contents={"highlights": True}`.
4. Dedupe by canonical URL or host.
5. Use Contents to crawl known promising URLs and targeted subpages.
6. Build compact evidence dossiers for Agent instead of sending huge raw page dumps.
7. Run Agent with a strict schema.
8. Run a second Agent audit or validation pass for high-value outputs.
9. Save raw inputs, raw outputs, structured results, costs, and prompts.
10. Export CSV/JSON/Markdown for review or frontend display.

For directory/list-building work, the best pattern from this project was:

```text
Search + Contents -> Agent curation -> Agent audit -> static JSON/CSV export -> human review UI
```

## Cost Notes

Always check current pricing before a large run: https://exa.ai/pricing

As of the docs checked during this project:

- Search base price: about `$7 / 1k requests` with up to 10 results.
- Deep Search: about `$12-15 / 1k requests`.
- Contents: about `$1 / 1k pages` per content type.
- Agent compute units: `$0.10 / ACU`.
- Agent search tool calls: `$0.005 / search`.
- Fixed Agent effort modes listed in docs:
  - `minimal`: `$0.012 / request`
  - `low`: `$0.025 / request`
  - `medium`: `$0.10 / request`
  - `high`: `$0.50 / request`
  - `xhigh`: `$1.00 / request`
- Email and phone contact enrichment are billed separately.

Practical cost controls:

- Start with 5-10 results before scaling.
- Prefer `highlights` over full `text`.
- Cap full text with `max_characters`.
- Dedupe aggressively before Agent.
- Use `maxItems` in schemas.
- Use fixed `effort` for predictable Agent runs.
- Batch row enrichment.
- Save known results and pass exclusions into later runs.
- Avoid asking for personal emails or phone numbers unless truly necessary.
- Print or save each run's `costDollars`/cost breakdown.

For Python SDK models, if unsure of exact field casing, inspect:

```python
print(completed.model_dump())
```

## Images

Exa can return image-related URLs, but it does not automatically create a rights-cleared image library.

Sources:

- Search results often include `image` and `favicon`.
- Search/Contents can request extracted image links via `extras.image_links` in Python SDK style.
- Contents can crawl subpages where images may appear.

Use images carefully:

- Keep the image source URL.
- Expect broken or hotlink-protected images.
- Do not assume reuse rights.
- Prefer official logos/favicons or generated/local assets when publication rights matter.

## Locations And Maps

Exa can help find public addresses, service areas, and contact pages. It does not replace a geocoder.

Recommended pattern:

1. Use Exa Search/Contents/Agent to find the official address or service area.
2. Classify whether an exact public pin is safe and appropriate.
3. Geocode safe exact addresses with a geocoder such as Nominatim, Mapbox, Google, or another provider.
4. Store precision separately from coordinates:

```text
exact_address
city
region
none
```

For sensitive services, do not show exact pins just because an address exists. Keep a review note explaining why the map precision was downgraded.

## Data Safety And Quality

Use Exa as a discovery and evidence tool, not as final authority.

General rules:

- Prefer official sources.
- Store evidence URLs.
- Keep human review for anything public-facing.
- Separate raw data from approved data.
- Track `last_verified_at`.
- Do not collect personal emails, phone numbers, or private addresses unless the use case genuinely requires it.
- Treat shelters, survivor services, private counseling, residential recovery, child/youth services, and similar sensitive contexts with extra caution.
- Mark low-confidence or conflicting evidence for manual review.

## Common Gotchas

- There is one Exa API key; Agent does not require a separate key.
- Python SDK uses snake_case; raw API and JavaScript use camelCase.
- On `/search`, `text`, `highlights`, and `summary` go inside `contents`.
- On `/contents`, `text`, `highlights`, and `summary` are top-level arguments.
- `use_autoprompt`/`useAutoprompt` is deprecated. Do not use it.
- Use `max_age_hours`/`maxAgeHours` instead of older livecrawl examples.
- `category="company"` and `category="people"` can disable some filters such as exclusions/date filters.
- Search result scores may not be present or meaningful for every search type.
- Agent runs are async; save the run id and poll until a terminal status.
- Contents responses may have per-URL failures; check statuses when using raw API responses.
- Do not assume Agent only uses your candidates unless you explicitly tell it to.

## Reusable Prompt Templates

Candidate curation:

```text
You are curating candidate records for a structured dataset.

Only evaluate candidates present in the input data. Do not add outside
entities. You may search the web only to verify, reject, enrich, or classify
the candidates.

For each accepted candidate, return the requested schema fields, source URLs,
confidence, and review notes. If evidence is missing or conflicting, mark the
candidate as manual_review rather than guessing.
```

Row enrichment:

```text
For each input row, find the requested public information using official
sources where possible. Return exactly one output item per input row. Preserve
the input id. If the information cannot be verified, leave the field blank or
null and explain the gap in review_notes.
```

Audit pass:

```text
Review the proposed records skeptically. Flag duplicates, unsupported claims,
out-of-scope records, stale URLs, missing evidence, unsafe publication choices,
and fields that require human review. Do not introduce new records in this
audit pass.
```

Deep search synthesis:

```text
Prefer official and primary sources. Avoid duplicate results. If sources
disagree, state the disagreement and assign lower confidence. Return structured
JSON that follows the schema exactly.
```

## Example Use Cases

- Build a directory of organizations, vendors, ministries, clinics, grants, schools, funds, or public resources.
- Enrich a CSV of companies with pricing pages, docs pages, executive names, hiring signals, or funding evidence.
- Monitor news about a topic and save source-backed summaries.
- Compare products or APIs from official docs and changelogs.
- Find and summarize recent research papers with citations.
- Build a lightweight research assistant that searches, reads pages, and returns grounded answers.
- Create a review UI from static JSON generated by an Exa pipeline.
- Find official location/contact data, then geocode with a separate mapping provider.

## Suggested File Outputs For Pipelines

For repeatable work, save artifacts like this:

```text
outputs/<task>/<timestamp>/
  queries.json
  raw_search_results.json
  candidates.json
  crawled_contents.json
  agent_prompt.txt
  agent_raw_run.json
  curated_results.json
  audit_raw_run.json
  audit_results.json
  summary.md
  cost.json
  export.csv
```

This makes it possible to resume work, compare iterations, exclude known results, debug Agent behavior, and show a human exactly where an output came from.

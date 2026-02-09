from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import json
import pickle
import base64
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from collections import defaultdict
from datetime import datetime
import re
from typing import List, Dict
import anthropic

app = FastAPI(title="Gmail Classifier API")

# --- Application matching helpers ---

PLATFORM_DOMAINS = {
    'linkedin.com', 'ashbyhq.com', 'myworkday.com', 'myworkdayjobs.com',
    'recruitee.com', 'greenhouse.io', 'lever.co', 'smartrecruiters.com',
    'icims.com', 'jobvite.com',
}

def _parse_sender(sender: str):
    """Parse 'Display Name <email@domain>' into (display_name, email_address)."""
    m = re.match(r'^(.*?)\s*<([^>]+)>', sender)
    if m:
        return m.group(1).strip().strip('"'), m.group(2).strip().lower()
    # bare email
    return '', sender.strip().lower()


def extract_company_key(email: dict) -> str:
    """Extract a normalized company identifier from an email for matching."""
    display_name, addr = _parse_sender(email.get('sender', ''))
    subject = email.get('subject', '')

    # Split address into local part and domain
    parts = addr.rsplit('@', 1)
    local = parts[0] if len(parts) == 2 else ''
    domain = parts[1] if len(parts) == 2 else addr

    # Check if this is a platform domain
    is_platform = any(domain.endswith(pd) for pd in PLATFORM_DOMAINS)

    if is_platform:
        if 'myworkday' in domain:
            # Use the email local part as company key (e.g. "Gartner@myworkday.com")
            return local.lower().strip()
        if 'linkedin.com' in domain:
            # Extract company from subject patterns like "sent to {Company}" or "at {Company}"
            for pat in [r'(?:sent to|at|with)\s+([A-Za-z0-9][\w\s&.\-]+?)(?:\s*[|!\-,]|\s+for\b|\s+in\b|$)',
                        r'^(.+?)\s+(?:is|has|sent)']:
                m = re.search(pat, subject, re.IGNORECASE)
                if m:
                    return m.group(1).strip().lower()
            # Fallback: use display name
            if display_name:
                return re.sub(r'\s+(via|on)\s+linkedin.*', '', display_name, flags=re.IGNORECASE).strip().lower()
            return 'linkedin_unknown'
        # Other platforms (ashbyhq, greenhouse, etc.) — use display name
        if display_name:
            # Strip common suffixes like "Recruitment Team", "Careers", "Hiring"
            cleaned = re.sub(r'\s+(recruitment\s+team|careers|hiring|talent|hr|jobs).*$', '',
                             display_name, flags=re.IGNORECASE).strip()
            return cleaned.lower()
        return local.lower().strip()

    # Non-platform: use domain, strip subdomains like mail., recruiting., broadcast., notifications.
    stripped = re.sub(r'^(mail|recruiting|broadcast|notifications|noreply|no-reply|careers|jobs|talent)\.',
                      '', domain)
    return stripped.lower()


def extract_job_ids(text: str) -> set:
    """Extract job reference numbers from subject/body text."""
    ids = set()
    # Pattern: job/req/ref/position/id followed by a number
    for m in re.finditer(r'(?:job|req|ref|position|id|requisition)\s*[:#\-]?\s*(\d{4,})', text, re.IGNORECASE):
        ids.add(m.group(1))
    # Pattern: #NUMBER
    for m in re.finditer(r'#(\d{4,})', text):
        ids.add(m.group(1))
    # Standalone 5+ digit numbers in subject line (likely job IDs)
    for m in re.finditer(r'\b(\d{5,})\b', text):
        ids.add(m.group(1))
    return ids


def _short_date(date_str: str) -> str:
    """Extract a short readable date like '7 Feb 2026' from an email Date header."""
    # Typical format: "Sat, 7 Feb 2026 05:54:19 +0000"
    m = re.search(r'(\d{1,2}\s+\w{3}\s+\d{4})', date_str)
    if m:
        return m.group(1)
    return date_str[:20] if date_str else 'unknown date'


def compute_application_matches(data: dict) -> dict:
    """Cross-reference applied emails against rejections/followups.

    Returns: { applied_email_id: { status: 'rejected'|'followup', matched_email_id, remark } }
    """
    JOB_CATEGORIES = {'job_applied', 'job_rejected', 'job_followup'}
    # Group emails by company key
    company_groups = defaultdict(lambda: defaultdict(list))
    for email in data.values():
        cat = email.get('category', '')
        if cat not in JOB_CATEGORIES:
            continue
        key = extract_company_key(email)
        company_groups[key][cat].append(email)

    matches = {}
    for company_key, groups in company_groups.items():
        applied_list = groups.get('job_applied', [])
        rejected_list = groups.get('job_rejected', [])
        followup_list = groups.get('job_followup', [])

        if not applied_list:
            continue
        if not rejected_list and not followup_list:
            continue

        # Simple case: exactly 1 applied email in this company group
        if len(applied_list) == 1:
            applied = applied_list[0]
            # Check rejections first
            if rejected_list:
                rej = rejected_list[0]
                matches[applied['id']] = {
                    'status': 'rejected',
                    'matched_email_id': rej['id'],
                    'remark': f"Rejection received on {_short_date(rej.get('date', ''))}",
                }
            elif followup_list:
                fu = followup_list[0]
                matches[applied['id']] = {
                    'status': 'followup',
                    'matched_email_id': fu['id'],
                    'remark': f"Followup on {_short_date(fu.get('date', ''))}",
                }
        else:
            # Multiple applied from same company — try to match by job IDs
            for applied in applied_list:
                applied_ids = extract_job_ids(applied.get('subject', ''))
                if not applied_ids:
                    continue  # No IDs → skip (conservative)

                for rej in rejected_list:
                    rej_ids = extract_job_ids(rej.get('subject', ''))
                    if applied_ids & rej_ids:
                        matches[applied['id']] = {
                            'status': 'rejected',
                            'matched_email_id': rej['id'],
                            'remark': f"Rejection received on {_short_date(rej.get('date', ''))}",
                        }
                        break

                if applied['id'] in matches:
                    continue

                for fu in followup_list:
                    fu_ids = extract_job_ids(fu.get('subject', ''))
                    if applied_ids & fu_ids:
                        matches[applied['id']] = {
                            'status': 'followup',
                            'matched_email_id': fu['id'],
                            'remark': f"Followup on {_short_date(fu.get('date', ''))}",
                        }
                        break

    return matches

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = os.path.join(os.path.dirname(__file__), 'analysis_data.json')

# --- Persistence helpers ---

def load_saved_data() -> dict:
    """Read persisted email classifications from disk."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_data(data: dict):
    """Write email classifications to disk."""
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def get_saved_summary(data: dict) -> dict:
    """Compute category stats and email list from saved data."""
    categories = defaultdict(list)
    all_emails = []
    for email in data.values():
        categories[email['category']].append(email)
        all_emails.append(email)

    total = len(all_emails)
    ignore_count = len(categories.get('ignore', []))

    category_stats = {}
    for cat, cat_emails in categories.items():
        if cat == 'ignore':
            continue
        category_stats[cat] = {
            'count': len(cat_emails),
            'percentage': (len(cat_emails) / total * 100) if total > 0 else 0,
        }

    matches = compute_application_matches(data)

    return {
        'categories': category_stats,
        'emails': all_emails,
        'total': total,
        'ignored': ignore_count,
        'matches': matches,
    }


def get_gmail_service():
    """Get authenticated Gmail service"""
    creds = None

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

    return build('gmail', 'v1', credentials=creds)

CLASSIFICATION_PROMPT = """You are an email classifier focused ONLY on job application emails. Classify each email into EXACTLY ONE of these categories:

- job_applied: Confirmation that a job application was received/submitted/acknowledged
- job_rejected: Rejection from a job application (not selected, position filled, moved forward with other candidates, etc.)
- job_followup: Follow-up communications about a job application YOU already submitted — interview invitations, assessments, offer letters, next steps, scheduling, status updates about YOUR candidacy (but not a straight rejection)
- ignore: Everything else — including job platform marketing/newsletters/alerts/recommendations, recruiter cold outreach pitching jobs you did NOT apply to, and ALL non-job emails

IMPORTANT RULES:
- job_followup is ONLY for emails about an application YOU already made. A recruiter or staffing agency reaching out to pitch a new role you never applied to is IGNORE
- Job platform newsletters, job alerts, "jobs for you" digests, job recommendations are IGNORE — they are marketing, not related to YOUR applications
- Only classify as job_applied/job_rejected/job_followup if the email is specifically about YOUR job application or candidacy
- Read the FULL email content carefully before classifying — do not rely on subject line alone
- An email about "not a match" from a non-job platform (e.g. ticket booking) is IGNORE

RESPONSE FORMAT:
Return a JSON array of strings, one per email, in the same order.
Example: ["ignore","job_applied","job_rejected","job_followup","ignore"]
"""


def get_claude_client():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    return anthropic.Anthropic(api_key=api_key)


def _strip_html(html):
    """Strip HTML tags and decode entities to get readable plain text."""
    # Remove style/script blocks
    text = re.sub(r'<(style|script)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove hidden elements
    text = re.sub(r'<[^>]+display\s*:\s*none[^>]*>.*?</[^>]+>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Replace block elements with newlines
    text = re.sub(r'<(br|/p|/div|/tr|/li|/h[1-6])[^>]*>', '\n', text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    text = text.replace('&middot;', '\u00b7').replace('&bull;', '\u2022')
    text = text.replace('&ndash;', '\u2013').replace('&mdash;', '\u2014')
    text = text.replace('&laquo;', '\u00ab').replace('&raquo;', '\u00bb')
    text = text.replace('&zwnj;', '').replace('&zwj;', '')
    text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
    text = re.sub(r'&\w+;', '', text)  # strip any remaining named entities
    # Collapse whitespace: spaces/tabs on each line, then blank lines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    # Remove lines that are only whitespace or single characters (artifacts)
    lines = text.split('\n')
    lines = [l.strip() for l in lines]
    text = '\n'.join(lines)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()


def clean_body_whitespace(body):
    """Collapse excessive whitespace and remove invisible Unicode junk."""
    # Remove zero-width and invisible Unicode characters
    text = re.sub(r'[\u200b\u200c\u200d\u034f\ufeff\u00ad]+', '', body)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    lines = [l.strip() for l in text.split('\n')]
    text = '\n'.join(lines)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()


def extract_email_body(msg):
    """Extract body from a Gmail message, preferring plain text, falling back to HTML."""
    payload = msg.get('payload', {})
    plain_text = ""
    html_text = ""

    def _decode(part):
        data = part.get('body', {}).get('data', '')
        if data:
            return base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
        return ''

    def _collect(part):
        nonlocal plain_text, html_text
        mime = part.get('mimeType', '')
        if mime == 'text/plain' and not plain_text:
            plain_text = _decode(part)
        elif mime == 'text/html' and not html_text:
            html_text = _decode(part)

    # Single-part message
    if 'parts' not in payload:
        _collect(payload)
    else:
        for part in payload.get('parts', []):
            _collect(part)
            for sub in part.get('parts', []):
                _collect(sub)

    # Decide between plain text and HTML.
    # Some senders (e.g. LinkedIn) put only footer/tracking URLs in text/plain
    # while the real content is in the HTML part. Always prefer whichever has
    # more meaningful (non-URL) content.
    html_body = ''
    if html_text:
        html_body = clean_body_whitespace(_strip_html(html_text))

    def _meaningful_length(text):
        """Length of text after removing URLs — measures actual content."""
        return len(re.sub(r'https?://\S+', '', text))

    if plain_text and html_body:
        plain_meaningful = _meaningful_length(plain_text)
        html_meaningful = _meaningful_length(html_body)
        if html_meaningful > plain_meaningful:
            return html_body
        return plain_text
    if plain_text:
        return plain_text
    if html_body:
        return html_body
    return msg.get('snippet', '')


def classify_batch(client, email_batch):
    """Classify a batch of emails using Claude. Returns list of dicts with 'category' and 'company'."""
    valid_categories = {'job_applied', 'job_rejected', 'job_followup', 'ignore'}
    fallback = {'category': 'ignore', 'company': None}

    # Build the prompt with email data
    emails_text = ""
    for i, e in enumerate(email_batch):
        body_preview = e['body'][:500] if e['body'] else e['snippet']
        emails_text += f"\n--- Email {i+1} ---\n"
        emails_text += f"From: {e['sender']}\n"
        emails_text += f"Subject: {e['subject']}\n"
        emails_text += f"Content: {body_preview}\n"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": CLASSIFICATION_PROMPT + "\n\nClassify these emails:\n" + emails_text
        }],
    )

    # Parse the response
    text = response.content[0].text.strip()
    print(f"[CLAUDE RAW] {text[:500]}", flush=True)
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        results = json.loads(match.group())
        parsed = []
        for r in results:
            if isinstance(r, dict):
                cat = r.get('category', 'ignore')
            else:
                cat = r if isinstance(r, str) else 'ignore'
            if cat not in valid_categories:
                cat = 'ignore'
            parsed.append({'category': cat, 'company': None})
        return parsed

    return [fallback] * len(email_batch)

@app.get("/")
def read_root():
    return {"message": "Gmail Classifier API", "status": "running"}


@app.get("/api/saved")
async def get_saved():
    """Return previously persisted classification data."""
    data = load_saved_data()
    if not data:
        return {'categories': {}, 'emails': [], 'total': 0, 'ignored': 0, 'matches': {}}
    return get_saved_summary(data)


@app.post("/api/backfill-bodies")
async def backfill_bodies():
    """Re-fetch full bodies for emails that only have short snippet text."""
    data = load_saved_data()
    short_ids = [eid for eid, e in data.items()
                 if len(e.get('body', '')) < 100
                 or e.get('body', '') == e.get('snippet', '')]
    if not short_ids:
        return {"updated": 0, "message": "All emails already have full bodies"}

    service = get_gmail_service()
    updated = 0
    for eid in short_ids:
        try:
            msg = service.users().messages().get(userId='me', id=eid, format='full').execute()
            body = extract_email_body(msg)
            data[eid]['body'] = body
            updated += 1
        except Exception as e:
            print(f"[BACKFILL ERROR] {eid}: {e}", flush=True)

    save_data(data)
    return {"updated": updated, "total_checked": len(short_ids)}


@app.post("/api/clean-bodies")
async def clean_bodies():
    """Clean up excessive whitespace in all stored email bodies."""
    data = load_saved_data()
    cleaned = 0
    for eid, e in data.items():
        body = e.get('body', '')
        if body:
            new_body = clean_body_whitespace(body)
            if new_body != body:
                data[eid]['body'] = new_body
                cleaned += 1
    save_data(data)
    return {"cleaned": cleaned, "total": len(data)}


@app.post("/api/reclassify-all")
async def reclassify_all():
    """Re-classify all saved emails using their current (full) bodies."""
    data = load_saved_data()
    if not data:
        return {"message": "No saved data to reclassify"}

    emails_list = list(data.values())
    BATCH_SIZE = 10
    client = get_claude_client()
    changes = 0

    for batch_start in range(0, len(emails_list), BATCH_SIZE):
        batch = emails_list[batch_start:batch_start + BATCH_SIZE]
        try:
            results = classify_batch(client, batch)
            for email_data, cls_result in zip(batch, results):
                new_cat = cls_result['category']
                old_cat = email_data['category']
                if new_cat != old_cat:
                    print(f"[RECLASS] {email_data['subject'][:50]}: {old_cat} -> {new_cat}", flush=True)
                    changes += 1
                data[email_data['id']]['category'] = new_cat
        except Exception as e:
            print(f"[RECLASS ERROR] batch {batch_start}: {e}", flush=True)

    save_data(data)
    summary = get_saved_summary(data)
    return {"changes": changes, "total": len(emails_list), **summary}


class ReclassifyRequest(BaseModel):
    email_id: str
    category: str


@app.post("/api/reclassify")
async def reclassify_email(req: ReclassifyRequest):
    """Move an email to a different category (or ignore it)."""
    valid_categories = {'job_applied', 'job_rejected', 'job_followup', 'ignore'}
    if req.category not in valid_categories:
        raise HTTPException(status_code=400, detail=f"Invalid category: {req.category}")

    data = load_saved_data()
    if req.email_id not in data:
        raise HTTPException(status_code=404, detail="Email not found in saved data")

    data[req.email_id]['category'] = req.category
    save_data(data)
    return get_saved_summary(data)


@app.get("/api/analyze")
async def analyze_emails(limit: int = None, after: str = None, before: str = None):
    """Analyze and classify emails with SSE progress updates.

    after/before: date strings in YYYY/MM/DD format.
    Defaults to current year if not provided.
    """

    def generate():
        try:
            service = get_gmail_service()

            # Build query for date range
            if not after:
                q_after = f"{datetime.now().year}/01/01"
            else:
                q_after = after
            if not before:
                q_before = f"{datetime.now().year + 1}/01/01"
            else:
                q_before = before
            q = f"after:{q_after} before:{q_before}"

            # Phase 1: collect all message IDs
            yield f"data: {json.dumps({'phase': 'listing', 'message': 'Fetching email list...'})}\n\n"

            messages = []
            page_token = None
            while True:
                kwargs = {
                    'userId': 'me',
                    'maxResults': 500,
                    'labelIds': ['INBOX'],
                    'q': q,
                }
                if page_token:
                    kwargs['pageToken'] = page_token
                results = service.users().messages().list(**kwargs).execute()
                messages.extend(results.get('messages', []))
                yield f"data: {json.dumps({'phase': 'listing', 'message': f'Found {len(messages)} emails so far...'})}\n\n"
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
                if limit and len(messages) >= limit:
                    messages = messages[:limit]
                    break

            total = len(messages)

            if total == 0:
                saved = load_saved_data()
                if saved:
                    summary = get_saved_summary(saved)
                    yield f"data: {json.dumps({'phase': 'done', 'result': summary})}\n\n"
                else:
                    yield f"data: {json.dumps({'phase': 'done', 'result': {'categories': {}, 'emails': [], 'total': 0, 'ignored': 0}})}\n\n"
                return

            # Filter out already-classified emails
            saved_data = load_saved_data()
            saved_ids = set(saved_data.keys())
            new_messages = [m for m in messages if m['id'] not in saved_ids]
            already_count = total - len(new_messages)

            if len(new_messages) == 0:
                yield f"data: {json.dumps({'phase': 'listing', 'message': f'All {total} emails already classified.'})}\n\n"
                summary = get_saved_summary(saved_data)
                yield f"data: {json.dumps({'phase': 'done', 'result': summary})}\n\n"
                return

            yield f"data: {json.dumps({'phase': 'analyzing', 'total': len(new_messages), 'processed': 0, 'message': f'{len(new_messages)} new emails to classify ({already_count} already classified)'})}\n\n"

            new_total = len(new_messages)

            # Phase 2: fetch full email content for NEW emails only
            fetched_emails = []
            for i, message in enumerate(new_messages):
                msg = service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='full'
                ).execute()

                headers = {h['name']: h['value'] for h in msg['payload']['headers']}
                subject = headers.get('Subject', 'No Subject')
                sender = headers.get('From', 'Unknown')
                date = headers.get('Date', '')
                snippet = msg.get('snippet', '')
                body = extract_email_body(msg)

                fetched_emails.append({
                    'id': message['id'],
                    'subject': subject,
                    'sender': sender,
                    'date': date,
                    'snippet': snippet,
                    'body': body,
                })

                if (i + 1) % 10 == 0 or (i + 1) == new_total:
                    yield f"data: {json.dumps({'phase': 'analyzing', 'total': new_total, 'processed': i + 1, 'important': 0, 'junk': 0, 'message': f'Fetched {i + 1} of {new_total} new emails...'})}\n\n"

            # Phase 3: classify in batches using Claude (new emails only)
            BATCH_SIZE = 10
            client = get_claude_client()
            classified = 0
            new_ignore = 0
            new_important = 0

            for batch_start in range(0, len(fetched_emails), BATCH_SIZE):
                batch = fetched_emails[batch_start:batch_start + BATCH_SIZE]
                try:
                    results = classify_batch(client, batch)
                except Exception as e:
                    print(f"[CLASSIFY ERROR] {e}", flush=True)
                    err_msg = str(e)
                    if 'credit balance' in err_msg.lower() or 'billing' in err_msg.lower():
                        yield f"data: {json.dumps({'phase': 'error', 'message': 'Anthropic API credit balance is too low. Please add credits at console.anthropic.com/settings/billing'})}\n\n"
                        return
                    results = [{'category': 'ignore', 'company': None}] * len(batch)

                for email_data, cls_result in zip(batch, results):
                    category = cls_result['category']

                    email_out = {
                        'id': email_data['id'],
                        'subject': email_data['subject'],
                        'sender': email_data['sender'],
                        'date': email_data['date'],
                        'snippet': email_data['snippet'],
                        'body': email_data['body'],
                        'category': category,
                    }
                    # Merge into saved data
                    saved_data[email_data['id']] = email_out

                    if category == 'ignore':
                        new_ignore += 1
                    else:
                        new_important += 1

                classified += len(batch)
                yield f"data: {json.dumps({'phase': 'classifying', 'total': new_total, 'processed': classified, 'important': new_important, 'ignored': new_ignore, 'message': f'Classified {classified} of {new_total} — {new_important} job-related, {new_ignore} ignored'})}\n\n"

            # Persist all data (saved + new)
            save_data(saved_data)

            # Phase 4: return ALL results (saved + new)
            summary = get_saved_summary(saved_data)
            yield f"data: {json.dumps({'phase': 'done', 'result': summary})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'phase': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/api/email/{email_id}")
async def get_email_details(email_id: str):
    """Get full email details"""

    try:
        service = get_gmail_service()

        msg = service.users().messages().get(
            userId='me',
            id=email_id,
            format='full'
        ).execute()

        headers = {h['name']: h['value'] for h in msg['payload']['headers']}

        # Get email body
        body = ""
        if 'parts' in msg['payload']:
            for part in msg['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    import base64
                    data = part['body'].get('data', '')
                    if data:
                        body = base64.urlsafe_b64decode(data).decode('utf-8')
                        break

        return {
            'id': email_id,
            'subject': headers.get('Subject', ''),
            'from': headers.get('From', ''),
            'to': headers.get('To', ''),
            'date': headers.get('Date', ''),
            'body': body or msg.get('snippet', '')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get email statistics"""

    try:
        service = get_gmail_service()

        # Get profile
        profile = service.users().getProfile(userId='me').execute()

        return {
            'email_address': profile['emailAddress'],
            'total_messages': profile['messagesTotal'],
            'total_threads': profile['threadsTotal']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

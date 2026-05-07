"""
CAPMeeting — Civil Air Patrol squadron weekly meeting agenda builder.

Squadron commanders (CCs) need to plan a 90-120 minute weekly meeting:
opening, safety briefing, AE moment, drill, character/leadership, squadron
business, closing. CC tells the app the date, who's attending, and the
current focus, and gets back a structured agenda with topic-specific
content for each block.

Public-domain content. No PII. No squadron rosters. Each request stateless.

Built by a CAP member as a free volunteer offering for squadron commanders.
"""
import collections
import functools
import json
import logging
import os
import re
import threading

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32))
app.config.update(
    SESSION_COOKIE_SECURE=os.environ.get('SESSION_COOKIE_SECURE', 'true').lower() == 'true',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('capmeeting')

_metrics = {'requests_total': 0, 'provider_success': collections.Counter(), 'provider_failure': collections.Counter()}
_metrics_lock = threading.Lock()


def _route_handler(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            logger.exception('Unhandled exception in %s', f.__name__)
            return jsonify(error='An error occurred. Please try again.'), 500
    return wrapper


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('X-Frame-Options', 'DENY')
    resp.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
    resp.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return resp


_HTTP_TIMEOUT = 35


def _llm_via_groq(system, user):
    key = os.environ.get('GROQ_KEY', '')
    if not key: return None
    r = requests.post('https://api.groq.com/openai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
              'temperature': 0.5, 'response_format': {'type': 'json_object'}},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_cerebras(system, user):
    key = os.environ.get('CEREBRAS_KEY', '')
    if not key: return None
    r = requests.post('https://api.cerebras.ai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('CEREBRAS_MODEL', 'llama-3.3-70b'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
              'temperature': 0.5},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_gemini(system, user):
    key = os.environ.get('GEMINI_KEY', '')
    if not key: return None
    r = requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}',
        headers={'Content-Type':'application/json'},
        json={'system_instruction':{'parts':[{'text':system}]},
              'contents':[{'role':'user','parts':[{'text':user}]}],
              'generationConfig':{'temperature':0.5, 'responseMimeType':'application/json'}},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['candidates'][0]['content']['parts'][0]['text']


def _llm_via_mistral(system, user):
    key = os.environ.get('MISTRAL_KEY', '')
    if not key: return None
    r = requests.post('https://api.mistral.ai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('MISTRAL_MODEL', 'mistral-small-latest'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
              'temperature': 0.5, 'response_format': {'type': 'json_object'}},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_huggingface(system, user):
    key = os.environ.get('HF_KEY', '')
    if not key: return None
    r = requests.post('https://router.huggingface.co/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('HF_MODEL', 'meta-llama/Llama-3.3-70B-Instruct'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
              'temperature': 0.5},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_sambanova(system, user):
    key = os.environ.get('SAMBANOVA_KEY', '')
    if not key: return None
    r = requests.post('https://api.sambanova.ai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('SAMBANOVA_MODEL', 'Meta-Llama-3.3-70B-Instruct'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}], 'temperature': 0.4},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_cloudflare(system, user):
    key = os.environ.get('CLOUDFLARE_AI_TOKEN', '')
    acct = os.environ.get('CLOUDFLARE_ACCOUNT_ID', '')
    if not key or not acct: return None
    model = os.environ.get('CLOUDFLARE_MODEL', '@cf/meta/llama-3.3-70b-instruct-fp8-fast')
    r = requests.post(f'https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}',
        headers={'Authorization': f'Bearer {key}'},
        json={'messages': [{'role':'system','content':system}, {'role':'user','content':user}], 'temperature': 0.4},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return j.get('result', {}).get('response') or j.get('result', {}).get('output') or ''

_PROVIDERS = [
    ('groq', _llm_via_groq),
    ('cerebras', _llm_via_cerebras),
    ('gemini', _llm_via_gemini),
    ('mistral', _llm_via_mistral),
    ('huggingface', _llm_via_huggingface),
    ('sambanova', _llm_via_sambanova),
    ('cloudflare', _llm_via_cloudflare),
]


def _llm(system: str, user: str) -> str:
    last_err = None
    for name, fn in _PROVIDERS:
        try:
            out = fn(system, user)
            if out:
                with _metrics_lock:
                    _metrics['provider_success'][name] += 1
                return out.strip()
        except Exception as e:
            last_err = e
            with _metrics_lock:
                _metrics['provider_failure'][name] += 1
            logger.warning('Provider %s failed: %s', name, e)
    raise RuntimeError(f'All LLM providers failed: {last_err}')


_MEETING_SYSTEM = (
    "You are a Civil Air Patrol squadron meeting agenda builder. The squadron commander gives you "
    "the meeting context (date, attendance mix, current focus, special considerations). You produce "
    "a complete 90-minute weekly meeting agenda with topic-specific content for each block.\n\n"
    "Output a JSON object with this structure:\n"
    '{\n'
    '  "meeting_title": "e.g., \'Squadron Meeting — 2026-05-08\'",\n'
    '  "total_minutes": 90,\n'
    '  "blocks": [\n'
    '    {\n'
    '      "name": "Opening Formation",\n'
    '      "minutes": 5,\n'
    '      "leader": "First Sergeant or Cadet Commander",\n'
    '      "content": "Specific script / instructions for this block",\n'
    '      "notes": "Optional additional notes"\n'
    '    }\n'
    '  ],\n'
    '  "safety_topic": "specific safety topic for this meeting (rotated, age-appropriate, current-season)",\n'
    '  "ae_moment": "specific aerospace education moment topic with 2-3 sentences of content",\n'
    '  "drill_focus": "specific drill objectives appropriate to the squadron skill mix",\n'
    '  "character_topic": "leadership / character development discussion topic with 2-3 prompt questions",\n'
    '  "business_items": ["list of suggested squadron business items based on focus area"],\n'
    '  "supplies_needed": ["physical materials the CC should have on hand"]\n'
    '}\n\n'
    "STANDARD BLOCKS (include these unless the CC asks for variation):\n"
    "1. Opening Formation (5 min) — Pledge of Allegiance, CAP Cadet Oath if cadets present, attendance call\n"
    "2. Safety Briefing (5 min) — rotating safety topic\n"
    "3. Aerospace Education Moment (5 min) — short AE topic, age-appropriate\n"
    "4. Drill (15 min) — practice appropriate to current skill level\n"
    "5. Character / Leadership (15 min) — discussion-based\n"
    "6. Squadron Business / Training (35 min) — main block, topic varies by focus area\n"
    "7. Closing Formation (5 min) — announcements, retreat ceremony, dismissal\n"
    "Total = 85 min agenda + 5 min buffer = 90 min meeting.\n\n"
    "RULES:\n"
    "- Output ONLY the JSON object. No prose around it.\n"
    "- Total meeting minutes must be 90 (or whatever the CC specifies in the input).\n"
    "- Safety topic must be timely (consider season — pollen, heat, holiday-period traffic, school year, etc.) and CAP-relevant (not generic OSHA).\n"
    "- AE moment must be substantive — not 'planes are cool'. Pick a specific concept and explain it briefly.\n"
    "- Drill focus must be calibrated to the cadet mix the CC describes (mostly new cadets = facing movements + marching basics; mostly experienced = formation transitions, manual of arms, ceremonial).\n"
    "- Character topic must be CAP-aligned: integrity, volunteer service, excellence, respect, or topical (DDR, ethics, leading peers).\n"
    "- Business items should reflect what the CC said is the current focus (e.g., 'SAREX prep' should generate appropriate logistical items: vehicle assignments, GTM3 training reminders, etc.).\n"
    "- Stay G-rated. Both cadets (12-21yo) and senior members are present.\n"
    "- Cite CAP regulations or pamphlets when relevant (CAPP 50-2 for senior leadership topics, CAPP 51-1 for cadet leadership lab activities).\n"
)


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith('```'):
        s = re.sub(r'^```[a-zA-Z]*\s*', '', s)
        s = re.sub(r'\s*```\s*$', '', s)
    return s.strip()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify(status='ok')


@app.route('/metrics')
def metrics():
    with _metrics_lock:
        return jsonify({
            'requests_total': _metrics['requests_total'],
            'provider_success': dict(_metrics['provider_success']),
            'provider_failure': dict(_metrics['provider_failure']),
        })


@app.route('/api/build', methods=['POST'])
@_route_handler
def build():
    data = request.get_json(silent=True) or {}
    meeting_date = (data.get('date') or '').strip()
    cadets = int(data.get('cadets') or 0)
    seniors = int(data.get('seniors') or 0)
    skill_mix = (data.get('skill_mix') or 'mixed').strip()
    focus = (data.get('focus') or '').strip()
    extra = (data.get('extra') or '').strip()
    minutes = max(60, min(120, int(data.get('minutes') or 90)))

    if not meeting_date:
        return jsonify(error='Pick a meeting date.'), 400

    user_msg = (
        f"MEETING CONTEXT:\n"
        f"- Date: {meeting_date}\n"
        f"- Attending: ~{cadets} cadets, ~{seniors} senior members\n"
        f"- Cadet skill mix: {skill_mix}\n"
        f"- Current focus area / upcoming activity: {focus or '(general training, no specific focus)'}\n"
        f"- Special considerations: {extra or '(none)'}\n"
        f"- Total meeting length: {minutes} minutes\n"
        f"\nGenerate a complete agenda."
    )

    with _metrics_lock:
        _metrics['requests_total'] += 1
    raw = _llm(_MEETING_SYSTEM, user_msg)
    raw = _strip_code_fence(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning('LLM returned non-JSON: %s', raw[:200])
        return jsonify(error='The model returned an unparseable agenda. Please try again.'), 502
    return jsonify(agenda=parsed)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

import copy
import json
import logging

import pytest

import app as application


VALID_REQUEST = {
    'date': '2026-08-11',
    'cadets': 12,
    'seniors': 6,
    'skill_mix': 'mixed',
    'focus': 'Emergency services training',
    'extra': 'Use de-identified examples only',
    'minutes': 90,
}

VALID_AGENDA = {
    'meeting_title': 'Squadron Meeting — 2026-08-11',
    'total_minutes': 90,
    'blocks': [
        {
            'name': 'Opening Formation',
            'minutes': 5,
            'leader': 'Cadet Commander',
            'content': 'Form the squadron and review the plan for the evening.',
            'notes': None,
        },
        {
            'name': 'Safety and Training',
            'minutes': 45,
            'leader': 'Assigned instructors',
            'content': 'Conduct the reviewed safety briefing and training activities.',
            'notes': 'Adjust activities to current local guidance.',
        },
        {
            'name': 'Squadron Business and Closing',
            'minutes': 35,
            'leader': 'Squadron Commander',
            'content': 'Review announcements, action items, and dismissal.',
            'notes': None,
        },
    ],
    'safety_topic': 'Review seasonal heat risk and hydration controls.',
    'ae_moment': 'Discuss how lift changes with airspeed and angle of attack.',
    'drill_focus': 'Practice formation transitions appropriate to a mixed group.',
    'character_topic': 'Discuss how integrity guides decisions when nobody is watching.',
    'business_items': ['Confirm upcoming training logistics'],
    'supplies_needed': ['Attendance worksheet', 'Training materials'],
}


@pytest.fixture
def client():
    application.app.config.update(TESTING=True)
    return application.app.test_client()


def test_public_routes_and_shared_security_headers(client):
    home = client.get('/')
    assert home.status_code == 200
    assert "frame-ancestors 'none'" in home.headers['Content-Security-Policy']
    assert home.headers['Permissions-Policy'].startswith('geolocation=()')
    assert client.get('/health').get_json() == {'status': 'ok'}
    assert client.get('/robots.txt').status_code == 200
    assert client.get('/sitemap.xml').status_code == 200


@pytest.mark.parametrize(
    'payload',
    [
        None,
        [],
        {},
        {**VALID_REQUEST, 'date': 'not-a-date'},
        {**VALID_REQUEST, 'cadets': '12'},
        {**VALID_REQUEST, 'seniors': -1},
        {**VALID_REQUEST, 'minutes': 75},
        {**VALID_REQUEST, 'skill_mix': 'expert'},
        {**VALID_REQUEST, 'focus': 'x' * 501},
        {**VALID_REQUEST, 'unexpected': True},
    ],
)
def test_build_rejects_invalid_requests(client, payload):
    if payload is None:
        response = client.post('/api/build', data='not-json')
    else:
        response = client.post('/api/build', json=payload)
    assert response.status_code == 400


def test_build_rejects_member_id_with_422_before_provider(client, monkeypatch):
    monkeypatch.setattr(
        application,
        '_llm',
        lambda *_args, **_kwargs: pytest.fail('provider must not be called'),
    )
    response = client.post(
        '/api/build',
        json={**VALID_REQUEST, 'focus': 'Coordinate CAPID 123456 for instruction'},
    )
    assert response.status_code == 422
    body = response.get_json()
    assert body['code'] == 'sensitive_data'
    assert body['detected_categories'] == ['member_id']


def test_build_rejects_ranked_member_name_before_provider(client, monkeypatch, caplog):
    marker = 'Lt Jane Doe'
    monkeypatch.setattr(
        application,
        '_llm',
        lambda *_args, **_kwargs: pytest.fail('provider must not be called'),
    )
    with caplog.at_level(logging.INFO, logger='capmeeting'):
        response = client.post(
            '/api/build',
            json={**VALID_REQUEST, 'extra': f'Ask {marker} to lead the activity'},
        )
    assert response.status_code == 422
    assert response.get_json()['detected_categories'] == ['member_name']
    assert marker not in caplog.text
    assert marker not in response.get_data(as_text=True)


def test_build_rejects_street_address_with_422(client, monkeypatch):
    monkeypatch.setattr(
        application,
        '_llm',
        lambda *_args, **_kwargs: pytest.fail('provider must not be called'),
    )
    response = client.post(
        '/api/build',
        json={**VALID_REQUEST, 'extra': 'Meet at 12 Maple Drive after formation'},
    )
    assert response.status_code == 422
    assert 'street_address' in response.get_json()['detected_categories']


def test_build_returns_validated_agenda_with_private_headers(client, monkeypatch):
    monkeypatch.setattr(
        application,
        '_llm',
        lambda *_args, **_kwargs: '```json\n' + json.dumps(VALID_AGENDA) + '\n```',
    )
    response = client.post('/api/build', json=VALID_REQUEST)
    assert response.status_code == 200
    assert response.get_json()['agenda'] == VALID_AGENDA
    assert response.headers['Cache-Control'] == 'private, no-store'
    assert response.headers['X-Robots-Tag'] == 'noindex, nofollow, noarchive'


def test_non_json_provider_output_is_not_logged(client, monkeypatch, caplog):
    marker = 'PRIVATE_AGENDA_OUTPUT_MARKER'
    monkeypatch.setattr(application, '_llm', lambda *_args, **_kwargs: marker)
    with caplog.at_level(logging.WARNING, logger='capmeeting'):
        response = client.post('/api/build', json=VALID_REQUEST)
    assert response.status_code == 502
    assert marker not in caplog.text
    assert marker not in response.get_data(as_text=True)
    assert 'llm_output_invalid reason=non_json' in caplog.text


def test_wrong_duration_or_extra_output_field_is_rejected(client, monkeypatch):
    invalid = {**VALID_AGENDA, 'total_minutes': 120, 'extra_field': 'not allowed'}
    monkeypatch.setattr(application, '_llm', lambda *_args, **_kwargs: json.dumps(invalid))
    response = client.post('/api/build', json=VALID_REQUEST)
    assert response.status_code == 502
    assert 'could not be validated' in response.get_json()['error']


def test_provider_output_with_identifier_is_rejected(client, monkeypatch, caplog):
    marker = 'Jane Doe'
    invalid = copy.deepcopy(VALID_AGENDA)
    invalid['blocks'][0]['content'] = f'Contact name is {marker} for instructions.'
    monkeypatch.setattr(application, '_llm', lambda *_args, **_kwargs: json.dumps(invalid))
    with caplog.at_level(logging.WARNING, logger='capmeeting'):
        response = client.post('/api/build', json=VALID_REQUEST)
    assert response.status_code == 502
    assert marker not in caplog.text
    assert marker not in response.get_data(as_text=True)


def test_provider_exception_text_and_traceback_are_not_logged(caplog):
    marker = 'PRIVATE_PROVIDER_EXCEPTION_MARKER'
    with caplog.at_level(logging.ERROR, logger='freshsky_common.llm'):
        try:
            raise RuntimeError(marker)
        except RuntimeError:
            logging.getLogger('freshsky_common.llm').exception(
                'LLM provider leaked %s', marker
            )
    assert marker not in caplog.text
    assert 'llm_provider_exception' in caplog.text


def test_metrics_and_policy_are_private_and_current(client):
    metrics = client.get('/metrics')
    assert metrics.headers['Cache-Control'] == 'private, no-store'
    assert metrics.headers['X-Robots-Tag'].startswith('noindex')
    privacy = client.get('/privacy').get_data(as_text=True)
    assert 'Last updated 2026-07-16' in privacy
    assert 'never meeting context or provider output' in privacy
    assert 'Google Gemini' not in privacy

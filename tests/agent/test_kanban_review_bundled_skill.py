"""Review workers survive an archived profile copy without relaxing skill policy."""
import json
from pathlib import Path

import pytest

from agent.skill_commands import _load_skill_payload, build_preloaded_skills_prompt


def test_review_preload_uses_bundled_contract_after_profile_archive(tmp_path, monkeypatch):
    from tools import skills_tool
    archived = tmp_path / '.archive' / 'sdlc-review'
    archived.mkdir(parents=True)
    (archived / 'SKILL.md').write_text('---\nname: sdlc-review\n---\nARCHIVED COPY')
    monkeypatch.setattr(skills_tool, 'SKILLS_DIR', tmp_path)
    monkeypatch.setenv('HERMES_KANBAN_TASK', 't_review')
    prompt, loaded, missing = build_preloaded_skills_prompt(['sdlc-review'])
    assert loaded == ['sdlc-review']
    assert missing == []
    assert 'kanban_request_changes' in prompt
    assert 'ARCHIVED COPY' not in prompt


def test_bundled_fallback_is_kanban_only(monkeypatch):
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    monkeypatch.setattr('tools.skills_tool.skill_view', lambda *a, **k: json.dumps(
        {'success': False, 'error': "Skill 'sdlc-review' not found."}))
    assert _load_skill_payload('sdlc-review') is None


@pytest.mark.parametrize('error', ['disabled', 'ambiguous', 'unsupported'])
def test_bundled_fallback_does_not_override_policy_errors(monkeypatch, error):
    monkeypatch.setenv('HERMES_KANBAN_TASK', 't_review')
    monkeypatch.setattr('tools.skills_tool.skill_view', lambda *a, **k: json.dumps(
        {'success': False, 'error': error}))
    assert _load_skill_payload('sdlc-review') is None


@pytest.mark.parametrize('disabled', ['sdlc-review', 'devops/sdlc-review'])
def test_archived_but_disabled_contract_stays_disabled(monkeypatch, disabled):
    monkeypatch.setenv('HERMES_KANBAN_TASK', 't_review')
    monkeypatch.setattr('tools.skills_tool.skill_view', lambda *a, **k: json.dumps(
        {'success': False, 'error': "Skill 'sdlc-review' not found."}))
    monkeypatch.setattr('agent.skill_utils.get_disabled_skill_names', lambda: {disabled})
    assert _load_skill_payload('sdlc-review') is None


def test_installed_contract_keeps_precedence(monkeypatch):
    monkeypatch.setenv('HERMES_KANBAN_TASK', 't_review')
    monkeypatch.setattr('tools.skills_tool.skill_view', lambda *a, **k: json.dumps(
        {'success': True, 'name': 'sdlc-review', 'content': 'PROFILE CONTRACT',
         'skill_dir': '/tmp/profile-review'}))
    loaded = _load_skill_payload('sdlc-review')
    assert loaded[0]['content'] == 'PROFILE CONTRACT'
    assert loaded[1] == Path('/tmp/profile-review')

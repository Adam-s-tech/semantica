"""Public checkpoint export and restore tests for TruthMaintenanceSession."""

import copy
import json

import pytest

from semantica.reasoning import (
    Derivation,
    FactSupport,
    Rule,
    TruthMaintenanceSession,
    TruthMaintenanceSnapshot,
)
from semantica.utils.exceptions import ProcessingError, ValidationError

CHECKPOINT_V1_JSON = """{
  "format_version": 1,
  "session_version": 4,
  "rules": [
    {
      "rule_id": "employment-eligibility",
      "conditions": ["Employed(?x)"],
      "conclusion": "Eligible(?x)"
    }
  ],
  "support_catalog": [
    {"support_id": "document-v1", "fact": "Employed(Alice)"},
    {"support_id": "document-v2", "fact": "Employed(Alice)"}
  ],
  "active_support_ids": ["document-v2"]
}"""

NORMALIZED_CHECKPOINT_V1 = {
    "format_version": 1,
    "session_version": 4,
    "rules": [
        {
            "rule_id": "employment-eligibility",
            "conditions": ["Employed(?x)"],
            "conclusion": "Eligible(?x)",
        }
    ],
    "support_catalog": [
        {"support_id": "document-v1", "fact": "Employed(Alice)"},
        {"support_id": "document-v2", "fact": "Employed(Alice)"},
    ],
    "active_support_ids": ["document-v2"],
}


def checkpoint_payload(rules, catalog, active, session_version):
    return {
        "format_version": 1,
        "session_version": session_version,
        "rules": [
            {
                "rule_id": rule_id,
                "conditions": list(conditions),
                "conclusion": conclusion,
            }
            for rule_id, conditions, conclusion in rules
        ],
        "support_catalog": [
            {"support_id": support_id, "fact": fact}
            for support_id, fact in sorted(catalog.items())
        ],
        "active_support_ids": sorted(active),
    }


def test_restore_preserves_two_source_withdrawals():
    original = TruthMaintenanceSession(
        rules=[Rule("eligibility", "Eligibility", ["Employed(?x)"], "Eligible(?x)")]
    )
    original.apply(
        assertions=[
            FactSupport("doc-1", "Employed(Alice)"),
            FactSupport("doc-2", "Employed(Alice)"),
        ]
    )
    restored = TruthMaintenanceSession.from_checkpoint(
        json.loads(json.dumps(original.to_checkpoint()))
    )

    assert restored.snapshot() == original.snapshot()
    assert restored.explain("Eligible(Alice)") == original.explain("Eligible(Alice)")
    for source in ("doc-1", "doc-2"):
        assert restored.apply(retractions=[source]) == original.apply(
            retractions=[source]
        )
        assert restored.snapshot() == original.snapshot()
        assert restored.explain("Eligible(Alice)") == original.explain(
            "Eligible(Alice)"
        )
    assert restored.facts == frozenset()


def test_checkpoint_survives_repeated_json_round_trips():
    original = TruthMaintenanceSession(
        rules=[Rule("employment", "Employment", ["Employed(?x)"], "Eligible(?x)")]
    )
    original.apply(
        assertions=[
            FactSupport("doc-1", "Employed(Alice)"),
            FactSupport("doc-2", "Employed(Alice)"),
        ]
    )
    payload = original.to_checkpoint()

    for _ in range(3):
        restored = TruthMaintenanceSession.from_checkpoint(
            json.loads(json.dumps(payload))
        )
        assert restored.snapshot() == original.snapshot()
        assert restored.explain("Eligible(Alice)") == original.explain(
            "Eligible(Alice)"
        )
        assert restored.to_checkpoint() == payload


def test_fixed_v1_fixture_restores_and_reexports_normalized_payload():
    restored = TruthMaintenanceSession.from_checkpoint(json.loads(CHECKPOINT_V1_JSON))

    assert restored.version == 4
    assert restored.snapshot() == TruthMaintenanceSnapshot(
        version=4,
        facts=frozenset({"Employed(Alice)", "Eligible(Alice)"}),
        active_supports=(FactSupport("document-v2", "Employed(Alice)"),),
    )
    assert restored.explain("Eligible(Alice)").derivations == (
        Derivation(
            rule_id="employment-eligibility",
            conclusion="Eligible(Alice)",
            premises=("Employed(Alice)",),
            bindings=(("?x", "Alice"),),
        ),
    )
    assert restored.to_checkpoint() == NORMALIZED_CHECKPOINT_V1


def test_empty_sessions_restore_with_version_zero():
    no_rules = TruthMaintenanceSession.from_checkpoint(
        {
            "format_version": 1,
            "session_version": 0,
            "rules": [],
            "support_catalog": [],
            "active_support_ids": [],
        }
    )
    assert no_rules.snapshot().version == 0
    assert no_rules.facts == frozenset()
    assert no_rules.to_checkpoint() == {
        "format_version": 1,
        "session_version": 0,
        "rules": [],
        "support_catalog": [],
        "active_support_ids": [],
    }

    rules_only = TruthMaintenanceSession.from_checkpoint(
        checkpoint_payload(
            rules=[("ab", ["A(?x)"], "B(?x)")],
            catalog={},
            active=[],
            session_version=0,
        )
    )
    assert rules_only.snapshot().version == 0
    assert rules_only.facts == frozenset()
    assert rules_only.to_checkpoint()["rules"] == [
        {"rule_id": "ab", "conditions": ["A(?x)"], "conclusion": "B(?x)"}
    ]


def test_all_supports_withdrawn_restores_catalog_and_positive_version():
    original = TruthMaintenanceSession(rules=[])
    original.apply(assertions=[FactSupport("retired", "Archived(Alice)")])
    original.apply(retractions=["retired"])
    payload = original.to_checkpoint()

    restored = TruthMaintenanceSession.from_checkpoint(copy.deepcopy(payload))

    assert restored.snapshot() == TruthMaintenanceSnapshot(
        version=2,
        facts=frozenset(),
        active_supports=(),
    )
    assert restored.to_checkpoint() == payload
    assert restored.to_checkpoint()["support_catalog"] == [
        {"support_id": "retired", "fact": "Archived(Alice)"}
    ]


def test_retired_support_binding_and_arity_survive_restore():
    original = TruthMaintenanceSession(rules=[])
    original.apply(assertions=[FactSupport("retired", "Q(a)")])
    original.apply(retractions=["retired"])
    restored = TruthMaintenanceSession.from_checkpoint(
        json.loads(json.dumps(original.to_checkpoint()))
    )

    for invalid_support in (
        FactSupport("retired", "Q(b)"),
        FactSupport("new-id", "Q(a, b)"),
    ):
        with pytest.raises(ValidationError):
            restored.apply(assertions=[invalid_support])

    assert restored.apply(
        assertions=[FactSupport("retired", "Q(a)")]
    ) == original.apply(assertions=[FactSupport("retired", "Q(a)")])
    assert restored.facts == frozenset({"Q(a)"})


def test_zero_arity_predicates_survive_restore():
    original = TruthMaintenanceSession(rules=[Rule("go", "Go", ["Ready()"], "Go()")])
    original.apply(assertions=[FactSupport("ready", "Ready()")])
    restored = TruthMaintenanceSession.from_checkpoint(
        json.loads(json.dumps(original.to_checkpoint()))
    )

    assert restored.snapshot() == original.snapshot()
    assert restored.explain("Go()") == original.explain("Go()")
    with pytest.raises(ValidationError):
        restored.apply(assertions=[FactSupport("bad", "Go(a)")])
    assert restored.apply(assertions=[FactSupport("go", "Go()")]) == original.apply(
        assertions=[FactSupport("go", "Go()")]
    )


def test_rule_only_predicate_arity_survives_restore_without_catalog_facts():
    original = TruthMaintenanceSession(rules=[Rule("ab", "AB", ["A(?x)"], "B(?x)")])
    restored = TruthMaintenanceSession.from_checkpoint(original.to_checkpoint())

    with pytest.raises(ValidationError):
        restored.apply(assertions=[FactSupport("bad-head", "B(a, b)")])
    with pytest.raises(ValidationError):
        restored.apply(assertions=[FactSupport("bad-body", "A(a, b)")])

    restored.apply(assertions=[FactSupport("source", "A(a)")])
    assert restored.facts == frozenset({"A(a)", "B(a)"})


@pytest.mark.parametrize(
    "payload",
    [
        checkpoint_payload(
            rules=[("bad", ["Not an atom"], "B(?x)")],
            catalog={},
            active=[],
            session_version=0,
        ),
        checkpoint_payload(
            rules=[("bad", ["A(?x)"], "Not an atom")],
            catalog={},
            active=[],
            session_version=0,
        ),
        checkpoint_payload(
            rules=[],
            catalog={"source": "Not an atom"},
            active=["source"],
            session_version=1,
        ),
        checkpoint_payload(
            rules=[],
            catalog={"source": "A(?x)"},
            active=["source"],
            session_version=1,
        ),
        checkpoint_payload(
            rules=[("unbound", ["A(?x)"], "B(?y)")],
            catalog={},
            active=[],
            session_version=0,
        ),
        checkpoint_payload(
            rules=[
                ("ab", ["A(?x)"], "B(?x)"),
                ("ba", ["B(?x)"], "A(?x)"),
            ],
            catalog={},
            active=[],
            session_version=0,
        ),
        checkpoint_payload(
            rules=[
                ("ab", ["A(?x)"], "B(?x)"),
                ("bc", ["B(?x)"], "C(?x)"),
                ("ca", ["C(?x)"], "A(?x)"),
            ],
            catalog={},
            active=[],
            session_version=0,
        ),
        checkpoint_payload(
            rules=[
                ("a1", ["A(?x)"], "B(?x)"),
                ("a2", ["A(?x, ?y)"], "C(?x, ?y)"),
            ],
            catalog={},
            active=[],
            session_version=0,
        ),
        checkpoint_payload(
            rules=[("ab", ["A(?x)"], "B(?x)")],
            catalog={"active": "A(a)", "inactive": "A(a, b)"},
            active=["active"],
            session_version=1,
        ),
    ],
    ids=[
        "invalid-condition",
        "invalid-conclusion",
        "invalid-fact",
        "variable-fact",
        "unbound-head-variable",
        "direct-cycle",
        "indirect-cycle",
        "rule-arity-conflict",
        "inactive-fact-arity-conflict",
    ],
)
def test_semantic_rejection_leaves_payload_unchanged(payload):
    before = copy.deepcopy(payload)
    with pytest.raises(ValidationError):
        TruthMaintenanceSession.from_checkpoint(payload)
    assert payload == before


def test_export_uses_captured_rules_not_caller_rule_mutation():
    caller_rule = Rule("captured", "Captured", ["A(?x)"], "B(?x)")
    session = TruthMaintenanceSession(rules=[caller_rule])

    caller_rule.rule_id = "mutated-id"
    caller_rule.name = "Mutated"
    caller_rule.conditions = ["Changed(?x)"]
    caller_rule.conclusion = "Changed(?x)"

    payload = session.to_checkpoint()
    assert payload["rules"] == [
        {"rule_id": "captured", "conditions": ["A(?x)"], "conclusion": "B(?x)"}
    ]

    restored = TruthMaintenanceSession.from_checkpoint(payload)
    restored.apply(assertions=[FactSupport("source", "A(a)")])
    assert restored.facts == frozenset({"A(a)", "B(a)"})


def test_payload_and_restored_session_are_isolated_in_both_directions():
    original = TruthMaintenanceSession(rules=[Rule("ab", "AB", ["A(?x)"], "B(?x)")])
    original.apply(assertions=[FactSupport("source", "A(a)")])
    expected_payload = original.to_checkpoint()
    before_snapshot = original.snapshot()
    payload = copy.deepcopy(expected_payload)
    restored = TruthMaintenanceSession.from_checkpoint(payload)

    payload["rules"][0]["rule_id"] = "mutated"
    payload["rules"][0]["conditions"].append("Injected(?x)")
    payload["support_catalog"][0]["fact"] = "Injected(a)"
    payload["active_support_ids"].append("source")

    assert original.to_checkpoint() == expected_payload
    assert original.snapshot() == before_snapshot
    assert restored.to_checkpoint() == expected_payload
    assert restored.snapshot() == before_snapshot

    restored.apply(assertions=[FactSupport("second", "A(b)")])
    assert payload == {
        "format_version": 1,
        "session_version": 1,
        "rules": [
            {
                "rule_id": "mutated",
                "conditions": ["A(?x)", "Injected(?x)"],
                "conclusion": "B(?x)",
            }
        ],
        "support_catalog": [
            {"support_id": "source", "fact": "Injected(a)"},
        ],
        "active_support_ids": ["source", "source"],
    }


def test_matcher_failure_is_wrapped_without_mutating_input_or_source(
    monkeypatch,
):
    original = TruthMaintenanceSession(rules=[Rule("ab", "AB", ["A(?x)"], "B(?x)")])
    original.apply(assertions=[FactSupport("source", "A(a)")])
    before_snapshot = original.snapshot()
    payload = original.to_checkpoint()
    before_payload = copy.deepcopy(payload)

    def fail_matcher(*args, **kwargs):
        raise RuntimeError("matcher unavailable")

    monkeypatch.setattr(TruthMaintenanceSession, "_match_rule", fail_matcher)

    assert original.to_checkpoint() == before_payload
    with pytest.raises(ProcessingError) as error:
        TruthMaintenanceSession.from_checkpoint(copy.deepcopy(payload))

    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "matcher unavailable"
    assert original.to_checkpoint() == before_payload
    assert original.snapshot() == before_snapshot
    assert payload == before_payload

    monkeypatch.undo()
    restored = TruthMaintenanceSession.from_checkpoint(payload)
    assert restored.snapshot() == before_snapshot

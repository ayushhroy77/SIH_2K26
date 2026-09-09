"""Test Suite for CVGuard Inference Plane (Phase 5).

Verifies:
1. Tamper with a stored record's `output` field after the fact (direct DB update, bypassing the API)
   — confirm /verify/{record_id} returns valid=false with a reason referencing the mismatch.
2. Attempt to replay an old valid record by reusing its sequence number for a new write
   — confirm this is rejected at write time (not silently accepted and only caught later).
3. Confirm a legitimate, untouched record verifies as valid=true.
4. Confirm GET /audit/verify (governance) still reports valid=true after this traffic.
5. Merkle tree construction, root computation, and audit proof verification.
6. Canonical hashing and atomic sequence number generation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import pytest
from httpx import ASGITransport, AsyncClient

from canonical import (
    canonical_json,
    canonical_record_payload,
    compute_config_hash,
    compute_input_hash,
    compute_output_hash,
    compute_record_hash,
)
from cvguard_schemas import AssetType, Finding, Severity, SignedFinding
from db import (
    ReplayAttackError,
    get_inference_record,
    get_next_sequence_number,
    reset_in_memory_state,
    save_inference_record,
    tamper_record_output,
)
from main import app
from merkle import MerkleTree, verify_merkle_proof


@pytest.fixture(autouse=True)
def clean_state():
    """Reset in-memory test state before every test."""
    reset_in_memory_state()
    from batcher import get_batch_manager
    bm = get_batch_manager()
    bm._pending_records.clear()
    bm._record_proofs.clear()
    yield
    reset_in_memory_state()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Provide a deterministic raw sample image byte buffer."""
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"


def test_canonical_binding_hashing(sample_image_bytes: bytes) -> None:
    """Ensure cryptographic leaf binding computation is deterministic and canonical."""
    model_id = "4a35b89a8123def4567890abcdef1234567890abcdef1234567890abcdef1234"
    config = {"resize": [224, 224], "norm": "imagenet", "temperature": 1.0}
    output = {"predicted_class": 3, "confidence": 0.942}

    input_hash = compute_input_hash(sample_image_bytes)
    config_hash = compute_config_hash(config)
    output_hash = compute_output_hash(output)
    seq = 1
    nonce = "abcd1234efgh5678"
    ts = "2026-09-09T00:00:00Z"

    rec_hash_1 = compute_record_hash(
        input_hash=input_hash,
        model_id=model_id,
        config_hash=config_hash,
        output=output,
        timestamp=ts,
        monotonic_sequence_no=seq,
        nonce=nonce,
    )
    rec_hash_2 = compute_record_hash(
        input_hash=input_hash,
        model_id=model_id,
        config_hash=config_hash,
        output=output,
        timestamp=ts,
        monotonic_sequence_no=seq,
        nonce=nonce,
    )

    assert rec_hash_1 == rec_hash_2
    assert len(rec_hash_1) == 64

    # Any modification to output must alter the binding digest
    rec_hash_modified = compute_record_hash(
        input_hash=input_hash,
        model_id=model_id,
        config_hash=config_hash,
        output={"predicted_class": 0, "confidence": 0.123},  # Tampered output
        timestamp=ts,
        monotonic_sequence_no=seq,
        nonce=nonce,
    )
    assert rec_hash_1 != rec_hash_modified


def test_atomic_sequence_numbers() -> None:
    """Ensure atomic sequence allocator generates monotonically increasing gapless numbers per model."""
    model_a = "model_alpha_digest_001"
    model_b = "model_beta_digest_002"

    assert get_next_sequence_number(model_a) == 1
    assert get_next_sequence_number(model_a) == 2
    assert get_next_sequence_number(model_a) == 3

    # Distinct models maintain distinct sequence streams
    assert get_next_sequence_number(model_b) == 1
    assert get_next_sequence_number(model_b) == 2
    assert get_next_sequence_number(model_a) == 4


def test_merkle_tree_construction_and_proofs() -> None:
    """Test Merkle tree root computation and proof verification across odd and even leaf counts."""
    for leaf_count in [1, 2, 3, 5, 8, 17]:
        leaves = [hashlib.sha256(f"leaf_{i}".encode()).hexdigest() for i in range(leaf_count)]
        tree = MerkleTree(leaves)

        assert tree.root is not None
        assert len(tree.root) == 64

        # Verify proof for every single leaf
        for idx in range(leaf_count):
            proof = tree.get_proof(idx)
            is_valid = verify_merkle_proof(
                leaf_hash=leaves[idx],
                proof=proof,
                expected_root=tree.root,
            )
            assert is_valid, f"Proof failed for leaf {idx} in tree of size {leaf_count}"

            # Tampering with leaf hash must invalidate proof
            assert not verify_merkle_proof(
                leaf_hash="a" * 64,
                proof=proof,
                expected_root=tree.root,
            )


# ==============================================================================
# 4 SPECIFIED TESTS FROM TASK 6
# ==============================================================================

@pytest.mark.asyncio
async def test_post_hoc_output_tamper_detected(
    sample_image_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK 6 TEST 1: Tamper with a stored record's `output` field after the fact
    (direct DB update, bypassing the API) — confirm /verify/{record_id} returns
    valid=false with a reason referencing the mismatch.
    """
    async def mock_dispatch_governance(finding: Finding, governance_url: str | None = None) -> SignedFinding:
        return SignedFinding(
            finding=finding,
            entry_hash="a" * 64,
            prev_hash="0" * 64,
            signature="b" * 128,
            ledger_id=101,
        )

    monkeypatch.setattr("batcher.dispatch_finding_to_governance", mock_dispatch_governance)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Ingest legitimate inference
        files = {"image": ("test.png", sample_image_bytes, "image/png")}
        data = {
            "model_id": "resnet50_tamper_test",
            "config": json.dumps({"task": "classification"}),
        }

        resp = await client.post("/infer", files=files, data=data)
        assert resp.status_code == 201, resp.text
        infer_data = resp.json()
        record_id = infer_data["record_id"]

        # 2. Flush batch so it is sealed
        flush_resp = await client.post("/batch/flush")
        assert flush_resp.status_code == 200

        # 3. Confirm legitimate record verifies as valid=true before tampering
        v_before = await client.get(f"/verify/{record_id}")
        assert v_before.status_code == 200
        assert v_before.json()["valid"] is True

        # 4. Perform direct DB update bypassing the API: tamper with the `output` field
        tampered_output = {
            "task": "classification",
            "top_class": "FORGED_LABEL_MALICIOUS",
            "confidence": 0.0001,
            "predictions": [{"label": "FORGED_LABEL_MALICIOUS", "score": 0.0001}],
            "execution_provider": "hacked_engine",
        }
        tamper_record_output(record_id=record_id, tampered_output=tampered_output)

        # 5. Call GET /verify/{record_id} — MUST return valid=False with reason referencing mismatch
        v_after = await client.get(f"/verify/{record_id}")
        assert v_after.status_code == 200
        result = v_after.json()

        assert result["valid"] is False, "Tampered output must not verify as valid"
        assert "tamper" in result["reason"].lower() or "mismatch" in result["reason"].lower() or "alteration" in result["reason"].lower(), (
            f"Expected reason referencing mismatch or alteration, got: {result['reason']}"
        )
        assert result["tampered"] is True
        assert result["status"] == "TAMPERED"


def test_replay_attack_rejected_at_write_time() -> None:
    """TASK 6 TEST 2: Attempt to replay an old valid record by reusing its sequence number
    for a new write — confirm this is rejected at write time (not silently accepted
    and only caught later).
    """
    model_id = "model_weight_digest_replay_test_12345"
    seq_no = 1

    # First write: legitimate initial sequence number
    save_inference_record(
        record_id="rec_initial_001",
        model_id=model_id,
        monotonic_sequence_no=seq_no,
        input_hash="input_hash_alpha_11111111111111111111111111111111111111111111111111111111",
        config_hash="config_hash_11111111111111111111111111111111111111111111111111111111",
        output_hash="output_hash_11111111111111111111111111111111111111111111111111111111",
        record_hash="record_hash_11111111111111111111111111111111111111111111111111111111",
        nonce="nonce_initial_1111",
        timestamp="2026-09-09T00:00:00Z",
        config_json={"setting": "original"},
        output_json={"prediction": "cat", "confidence": 0.99},
    )

    # Second write: Adversary attempts to replay the same sequence number (seq_no = 1) for a new write
    # MUST be rejected at write time via ReplayAttackError (violates uq_model_sequence constraint)
    with pytest.raises(ReplayAttackError) as exc_info:
        save_inference_record(
            record_id="rec_replayed_002",
            model_id=model_id,
            monotonic_sequence_no=seq_no,  # Replay!
            input_hash="input_hash_beta_22222222222222222222222222222222222222222222222222222222",
            config_hash="config_hash_22222222222222222222222222222222222222222222222222222222",
            output_hash="output_hash_22222222222222222222222222222222222222222222222222222222",
            record_hash="record_hash_22222222222222222222222222222222222222222222222222222222",
            nonce="nonce_replay_2222",
            timestamp="2026-09-09T00:01:00Z",
            config_json={"setting": "replayed"},
            output_json={"prediction": "dog", "confidence": 0.88},
        )

    assert "REPLAY ATTACK REJECTED" in str(exc_info.value)
    assert f"Sequence number {seq_no} for model {model_id} already exists" in str(exc_info.value)


@pytest.mark.asyncio
async def test_legitimate_record_verifies_valid(
    sample_image_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK 6 TEST 3: Confirm a legitimate, untouched record verifies as valid=true."""
    async def mock_dispatch_governance(finding: Finding, governance_url: str | None = None) -> SignedFinding:
        return SignedFinding(
            finding=finding,
            entry_hash="c" * 64,
            prev_hash="0" * 64,
            signature="d" * 128,
            ledger_id=202,
        )

    monkeypatch.setattr("batcher.dispatch_finding_to_governance", mock_dispatch_governance)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Ingest legitimate inference
        files = {"image": ("clean.png", sample_image_bytes, "image/png")}
        data = {
            "model_id": "resnet50_clean_check",
            "config": json.dumps({"task": "detection", "threshold": 0.5}),
        }

        resp = await client.post("/infer", files=files, data=data)
        assert resp.status_code == 201
        record_id = resp.json()["record_id"]

        # Flush into Merkle batch
        flush_resp = await client.post("/batch/flush")
        assert flush_resp.status_code == 200

        # Query GET /verify/{record_id}
        verify_resp = await client.get(f"/verify/{record_id}")
        assert verify_resp.status_code == 200
        res = verify_resp.json()

        assert res["valid"] is True, f"Legitimate record should be valid. Reason: {res.get('reason')}"
        assert res["tampered"] is False
        assert res["proof_valid"] is True
        assert res["governance_sealed"] is True
        assert res["ledger_id"] == 202
        assert "Cryptographically verified" in res["reason"]


@pytest.mark.asyncio
async def test_governance_audit_verify_after_traffic(
    sample_image_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK 6 TEST 4: Confirm GET /audit/verify (governance) still reports valid=true after this traffic."""
    # We construct a real in-memory governance AuditLedger and LocalFileSigner to execute real cryptographic hash chaining
    import sys
    sys.path.insert(0, "/cvguard/services/governance")
    from ledger import AuditLedger
    from signer import LocalFileSigner

    signer = LocalFileSigner()
    governance_ledger = AuditLedger(signer=signer)

    async def real_governance_append(finding: Finding, governance_url: str | None = None) -> SignedFinding:
        # Directly invoke the governance ledger's real append_entry logic
        return governance_ledger.append_entry(finding)

    monkeypatch.setattr("batcher.dispatch_finding_to_governance", real_governance_append)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Submit 3 inference queries across multiple models
        for i in range(3):
            files = {"image": (f"img_{i}.png", sample_image_bytes + bytes([i]), "image/png")}
            data = {"model_id": f"model_traffic_{i % 2}", "config": json.dumps({"idx": i})}
            resp = await client.post("/infer", files=files, data=data)
            assert resp.status_code == 201

        # Flush batch into governance ledger
        flush_resp = await client.post("/batch/flush")
        assert flush_resp.status_code == 200
        assert flush_resp.json()["size"] == 3

        # Execute GET /audit/verify on the governance ledger
        audit_result = governance_ledger.verify_chain()
        assert audit_result.valid is True, f"Governance chain audit failed: {audit_result.reason}"
        assert audit_result.total_entries >= 1
        assert audit_result.corrupted_entry_id is None
        assert "verified successfully" in audit_result.reason.lower()

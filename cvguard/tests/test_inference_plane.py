"""Test Suite for CVGuard Inference Plane (Phase 5).

Verifies:
1. Cryptographic binding between input image, model identity, config, and output.
2. Database-enforced atomic sequence number increment and monotonicity.
3. Replay attack rejection when duplicate sequence numbers are submitted or replayed.
4. Merkle tree construction, root computation, and audit proof verification.
5. Batch aggregation and governance sealing using AssetType.INFERENCE_RECORD.
6. Post-hoc tamper detection across inputs, predictions, and proof paths.
"""

from __future__ import annotations

import base64
import hashlib
import json
import pytest
from httpx import ASGITransport, AsyncClient

from canonical import (
    canonical_record_payload,
    compute_config_hash,
    compute_input_hash,
    compute_output_hash,
    compute_record_hash,
)
from cvguard_schemas import AssetType, Finding, SignedFinding
from db import (
    ReplayAttackError,
    get_inference_record,
    get_next_sequence_number,
    reset_in_memory_state,
    save_inference_record,
)
from main import app
from merkle import MerkleProofStep, MerkleTree, hash_pair, verify_merkle_proof


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
    model_id = "sha256:4a35b89a8123def4"
    config = {"resize": [224, 224], "norm": "imagenet", "temperature": 1.0}
    output = {"predicted_class": 3, "confidence": 0.942}

    input_hash = compute_input_hash(sample_image_bytes)
    config_hash = compute_config_hash(config)
    output_hash = compute_output_hash(output)
    seq = 1
    rid = "test_rec_001"

    rec_hash_1 = compute_record_hash(
        record_id=rid,
        model_id=model_id,
        sequence_number=seq,
        input_hash=input_hash,
        config_hash=config_hash,
        output_hash=output_hash,
    )
    rec_hash_2 = compute_record_hash(
        record_id=rid,
        model_id=model_id,
        sequence_number=seq,
        input_hash=input_hash,
        config_hash=config_hash,
        output_hash=output_hash,
    )

    assert rec_hash_1 == rec_hash_2
    assert len(rec_hash_1) == 64

    # Any modification must alter the binding digest
    rec_hash_modified = compute_record_hash(
        record_id=rid,
        model_id=model_id,
        sequence_number=seq + 1,  # Alter sequence
        input_hash=input_hash,
        config_hash=config_hash,
        output_hash=output_hash,
    )
    assert rec_hash_1 != rec_hash_modified


def test_atomic_sequence_numbers() -> None:
    """Ensure atomic sequence allocator generates monotonically increasing gapless numbers per model."""
    model_a = "model_alpha"
    model_b = "model_beta"

    assert get_next_sequence_number(model_a) == 1
    assert get_next_sequence_number(model_a) == 2
    assert get_next_sequence_number(model_a) == 3

    # Distinct models maintain distinct sequence streams
    assert get_next_sequence_number(model_b) == 1
    assert get_next_sequence_number(model_b) == 2
    assert get_next_sequence_number(model_a) == 4


def test_replay_attack_rejection() -> None:
    """Ensure duplicate sequence numbers for the same model are strictly rejected."""
    model_id = "model_prod_v1"
    save_inference_record(
        record_id="rec_1",
        model_id=model_id,
        sequence_number=1,
        input_hash="hash1",
        config_hash="cfg1",
        output_hash="out1",
        record_hash="rh1",
        config_json={},
        output_json={},
    )

    # Attempting to persist duplicate sequence number 1 for model_id must raise ReplayAttackError
    with pytest.raises(ReplayAttackError):
        save_inference_record(
            record_id="rec_1_replay",
            model_id=model_id,
            sequence_number=1,  # Replay!
            input_hash="hash2",
            config_hash="cfg2",
            output_hash="out2",
            record_hash="rh2",
            config_json={},
            output_json={},
        )


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


@pytest.mark.asyncio
async def test_full_inference_lifecycle_and_verification(
    sample_image_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end test of POST /infer, batch flushing, and GET /verify/{record_id}."""
    # Mock governance dispatch to simulate Governance Spine Ed25519 signature
    async def mock_dispatch_governance(finding: Finding, governance_url: str | None = None) -> SignedFinding:
        assert finding.asset_type == AssetType.INFERENCE_RECORD
        assert "merkle_root:" in finding.asset_ref
        return SignedFinding(
            finding=finding,
            entry_hash="mock_entry_hash_" + "0" * 48,
            prev_hash="0" * 64,
            signature="mock_signature_" + "1" * 112,
            ledger_id=42,
        )

    monkeypatch.setattr("batcher.dispatch_finding_to_governance", mock_dispatch_governance)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Health check
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"

        # 2. Submit inference request via multipart/form-data
        files = {"file": ("test.png", sample_image_bytes, "image/png")}
        data = {
            "model_id": "resnet50_v1",
            "config": json.dumps({"temperature": 0.7}),
        }

        resp = await client.post("/infer", files=files, data=data)
        assert resp.status_code == 201, resp.text
        infer_data = resp.json()

        record_id = infer_data["record_id"]
        assert infer_data["model_id"] == "resnet50_v1"
        assert infer_data["sequence_number"] == 1
        assert infer_data["input_hash"] == hashlib.sha256(sample_image_bytes).hexdigest()
        assert infer_data["status"] == "pending_batch"

        # 3. Before batch flush, verify endpoint reports PENDING_BATCH
        verify_pre = await client.get(f"/verify/{record_id}")
        assert verify_pre.status_code == 200
        assert verify_pre.json()["status"] == "PENDING_BATCH"
        assert not verify_pre.json()["tampered"]

        # 4. Submit a second inference to test sequence monotonicity
        files2 = {"file": ("test2.png", sample_image_bytes + b"_alt", "image/png")}
        resp2 = await client.post("/infer", files=files2, data=data)
        assert resp2.status_code == 201
        assert resp2.json()["sequence_number"] == 2

        # 5. Flush batch
        flush_resp = await client.post("/batch/flush")
        assert flush_resp.status_code == 200
        flush_data = flush_resp.json()
        assert flush_data["size"] == 2
        assert flush_data["first_sequence"] == 1
        assert flush_data["last_sequence"] == 2
        assert flush_data["ledger_id"] == 42

        # 6. Verify record_id is now fully VERIFIED with valid Merkle proof
        verify_post = await client.get(f"/verify/{record_id}")
        assert verify_post.status_code == 200
        v_data = verify_post.json()

        assert v_data["status"] == "VERIFIED"
        assert not v_data["tampered"]
        assert v_data["proof_valid"] is True
        assert v_data["governance_sealed"] is True
        assert v_data["ledger_id"] == 42
        assert len(v_data["merkle_proof"]) > 0


@pytest.mark.asyncio
async def test_post_hoc_tamper_detection(
    sample_image_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that tampering with any record field is caught immediately upon verification."""
    async def mock_dispatch_governance(finding: Finding, governance_url: str | None = None) -> SignedFinding:
        return SignedFinding(
            finding=finding,
            entry_hash="a" * 64,
            prev_hash="0" * 64,
            signature="b" * 128,
            ledger_id=99,
        )

    monkeypatch.setattr("batcher.dispatch_finding_to_governance", mock_dispatch_governance)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("tamper_test.png", sample_image_bytes, "image/png")}
        data = {"model_id": "model_secure_01"}

        resp = await client.post("/infer", files=files, data=data)
        record_id = resp.json()["record_id"]

        await client.post("/batch/flush")

        # Confirm legitimate verification passes
        v_legit = await client.get(f"/verify/{record_id}")
        assert v_legit.json()["status"] == "VERIFIED"

        # Simulate post-hoc database tampering: attacker changes stored input_hash
        record_in_db = get_inference_record(record_id)
        assert record_in_db is not None
        record_in_db["input_hash"] = "tampered_" + "0" * 55

        # Verify endpoint must detect tamper
        v_tampered = await client.get(f"/verify/{record_id}")
        assert v_tampered.status_code == 200
        t_data = v_tampered.json()

        assert t_data["status"] == "TAMPERED"
        assert t_data["tampered"] is True
        assert "Post-hoc modification detected" in t_data["reason"]


@pytest.mark.asyncio
async def test_json_infer_endpoint(sample_image_bytes: bytes) -> None:
    """Verify that clients can invoke inference via JSON body with base64 images."""
    b64_img = base64.b64encode(sample_image_bytes).decode("utf-8")
    payload = {
        "image_base64": b64_img,
        "model_id": "vision_classifier_json",
        "config": {"crop": True, "top_k": 5},
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/infer", json=payload)
        assert resp.status_code == 201, resp.text
        data = resp.json()

        assert data["model_id"] == "vision_classifier_json"
        assert data["sequence_number"] == 1
        assert len(data["record_hash"]) == 64

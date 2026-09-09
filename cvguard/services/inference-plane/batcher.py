"""Batch management, Merkle tree aggregation, and governance sealing for CVGuard Inference Plane."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from cvguard_schemas import AssetType, Disposition, Finding, Severity, SignedFinding
from db import (
    get_inference_batch,
    get_inference_record,
    save_inference_batch,
    update_records_batch_id,
)
from governance_client import dispatch_finding_to_governance
from merkle import MerkleProofStep, MerkleTree

logger = logging.getLogger("cvguard.inferenceplane.batcher")

BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "100"))
BATCH_FLUSH_INTERVAL_SECONDS = float(os.getenv("BATCH_FLUSH_INTERVAL_SECONDS", "5.0"))


class BatchManager:
    """Orchestrates buffering of inference records, Merkle tree batching, and governance dispatch."""

    def __init__(self, max_batch_size: int = BATCH_MAX_SIZE) -> None:
        self.max_batch_size = max_batch_size
        self._lock = asyncio.Lock()
        self._pending_records: list[dict[str, Any]] = []
        # In-memory index of record_id -> (batch_id, leaf_index, proof) for fast verification lookup
        self._record_proofs: dict[str, dict[str, Any]] = {}

    @property
    def pending_count(self) -> int:
        """Count of records buffered waiting to be sealed."""
        return len(self._pending_records)

    async def add_record(self, record_data: dict[str, Any]) -> str | None:
        """Add an inference record to the pending buffer.

        If buffer reaches `max_batch_size`, automatically triggers batch flush.
        Returns batch_id if flushed immediately, None if still buffered.
        """
        should_flush = False
        async with self._lock:
            self._pending_records.append(record_data)
            if len(self._pending_records) >= self.max_batch_size:
                should_flush = True

        if should_flush:
            batch = await self.flush()
            return batch.get("batch_id") if batch else None

        return None

    async def flush(self) -> dict[str, Any] | None:
        """Flush pending records into a cryptographic Merkle batch and seal via Governance Spine."""
        async with self._lock:
            if not self._pending_records:
                return None

            items = list(self._pending_records)
            self._pending_records.clear()

        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        leaf_hashes = [item["record_hash"] for item in items]
        record_ids = [item["record_id"] for item in items]

        # 1. Build Merkle tree over record binding hashes
        merkle_tree = MerkleTree(leaf_hashes)
        merkle_root = merkle_tree.root

        first_seq = items[0].get("monotonic_sequence_no", items[0].get("sequence_number", 1))
        last_seq = items[-1].get("monotonic_sequence_no", items[-1].get("sequence_number", 1))
        model_id = items[0]["model_id"]

        logger.info(
            "Creating Merkle batch %s (%d records, sequences %d..%d, root: %s...)",
            batch_id,
            len(items),
            first_seq,
            last_seq,
            merkle_root[:16],
        )

        # 2. Precompute and index Merkle proofs for all records in this batch
        for idx, item in enumerate(items):
            rid = item["record_id"]
            proof = merkle_tree.get_proof(idx)
            self._record_proofs[rid] = {
                "batch_id": batch_id,
                "leaf_index": idx,
                "merkle_root": merkle_root,
                "proof": proof,
            }

        # 3. Formulate canonical Finding with AssetType.INFERENCE_RECORD
        finding = Finding(
            asset_type=AssetType.INFERENCE_RECORD,
            asset_ref=f"merkle_root:{merkle_root}",
            detector="cvguard.inferenceplane.merkle_batcher:v1.0",
            reason=(
                f"Cryptographically verified inference batch {batch_id} comprising {len(items)} records "
                f"sealed with Merkle root {merkle_root[:16]}."
            ),
            evidence=[
                f"batch_id:{batch_id}",
                f"merkle_root:{merkle_root}",
                f"record_count:{len(items)}",
                f"first_sequence:{first_seq}",
                f"last_sequence:{last_seq}",
                f"model_id:{model_id}",
            ],
            confidence=1.0,
            severity=Severity.INFO,
            disposition=Disposition.ACCEPT,
            assumptions=[
                "Merkle tree leaves computed from canonical JSON binding payloads.",
                "Atomic database sequence number monotonic continuity enforced.",
            ],
            limitations=[
                "Verification requires access to published Merkle root in governance ledger.",
            ],
        )

        # 4. Dispatch finding to Governance Spine to obtain Ed25519 signature and ledger sequence
        try:
            signed_finding: SignedFinding = await dispatch_finding_to_governance(finding)
            ledger_id = signed_finding.ledger_id
            signature = signed_finding.signature
            signed_dict = signed_finding.model_dump(mode="json")
        except Exception as exc:
            logger.error("Governance dispatch failed for batch %s: %s", batch_id, exc)
            # Re-queue items if governance failed so they are not lost
            async with self._lock:
                self._pending_records = items + self._pending_records
            raise

        # 5. Persist batch in database
        save_inference_batch(
            batch_id=batch_id,
            merkle_root=merkle_root,
            size=len(items),
            first_sequence=first_seq,
            last_sequence=last_seq,
            model_id=model_id,
            finding_id=finding.finding_id,
            ledger_id=ledger_id,
            signed_finding=signed_dict,
        )

        # 6. Update individual records in database with batch_id
        update_records_batch_id(record_ids=record_ids, batch_id=batch_id)

        logger.info(
            "Batch %s sealed into Governance Ledger ID %s with root %s",
            batch_id,
            ledger_id,
            merkle_root[:16],
        )

        return {
            "batch_id": batch_id,
            "size": len(items),
            "first_sequence": first_seq,
            "last_sequence": last_seq,
            "merkle_root": merkle_root,
            "finding_id": finding.finding_id,
            "ledger_id": ledger_id,
            "signature": signature,
            "signed_finding": signed_dict,
        }

    def get_proof_for_record(self, record_id: str) -> tuple[dict[str, Any] | None, list[MerkleProofStep]]:
        """Retrieve containing batch and Merkle proof for a specific record_id."""
        cached = self._record_proofs.get(record_id)
        if cached:
            batch = get_inference_batch(cached["batch_id"])
            return batch, cached["proof"]

        # If not in cache, check if record is in DB with batch_id
        rec = get_inference_record(record_id)
        if not rec or not rec.get("batch_id"):
            return None, []

        batch = get_inference_batch(rec["batch_id"])
        # If batch size is 1, proof is empty list (root == leaf)
        if batch and batch.get("size") == 1:
            return batch, []

        return batch, []


# Global singleton batch manager instance
_batch_manager: BatchManager | None = None


def get_batch_manager() -> BatchManager:
    """Accessor for the active BatchManager singleton."""
    global _batch_manager
    if _batch_manager is None:
        _batch_manager = BatchManager()
    return _batch_manager

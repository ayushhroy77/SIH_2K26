"""Cryptographic Merkle Tree construction and proof validation for CVGuard Inference Plane.

Enables verifiable batching of individual inference record binding hashes into a single
cryptographic root that is sealed into the Governance Spine ledger.
"""

from __future__ import annotations

import hashlib
from typing import Literal
from pydantic import BaseModel, Field


class MerkleProofStep(BaseModel):
    """Atomic step in a Merkle verification path from leaf to root."""

    sibling_hash: str = Field(..., description="64-character hex digest of the sibling node.")
    position: Literal["left", "right"] = Field(
        ...,
        description="Position of the sibling relative to the accumulated path node ('left' or 'right').",
    )


def hash_pair(left_hex: str, right_hex: str) -> str:
    """Compute parent hash from ordered sibling pair: SHA-256(left_hex || right_hex)."""
    combined = (left_hex + right_hex).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


class MerkleTree:
    """Deterministic binary Merkle Tree built over an ordered list of leaf hashes."""

    def __init__(self, leaves: list[str]) -> None:
        if not leaves:
            raise ValueError("Cannot construct MerkleTree with empty leaf list.")
        self.leaves: list[str] = list(leaves)
        self.levels: list[list[str]] = self._build_levels()

    def _build_levels(self) -> list[list[str]]:
        """Construct bottom-up tree levels up to the root."""
        levels: list[list[str]] = [self.leaves]
        current = self.leaves

        while len(current) > 1:
            next_level: list[str] = []
            working = list(current)
            if len(working) % 2 == 1:
                # Duplicate last leaf for odd-length level
                working.append(working[-1])

            for i in range(0, len(working), 2):
                parent = hash_pair(working[i], working[i + 1])
                next_level.append(parent)

            levels.append(next_level)
            current = next_level

        return levels

    @property
    def root(self) -> str:
        """64-character hex SHA-256 root digest of the Merkle Tree."""
        return self.levels[-1][0]

    def get_proof(self, leaf_index: int) -> list[MerkleProofStep]:
        """Generate audit proof (path of sibling hashes) for leaf at `leaf_index`."""
        if leaf_index < 0 or leaf_index >= len(self.leaves):
            raise IndexError(f"leaf_index {leaf_index} out of bounds (0..{len(self.leaves) - 1})")

        proof: list[MerkleProofStep] = []
        idx = leaf_index

        # Iterate through levels up to root level (excluding root level itself)
        for level in self.levels[:-1]:
            working = list(level)
            if len(working) % 2 == 1:
                working.append(working[-1])

            if idx % 2 == 0:
                # Target is left child, sibling is right
                sibling_hash = working[idx + 1]
                proof.append(MerkleProofStep(sibling_hash=sibling_hash, position="right"))
            else:
                # Target is right child, sibling is left
                sibling_hash = working[idx - 1]
                proof.append(MerkleProofStep(sibling_hash=sibling_hash, position="left"))

            idx = idx // 2

        return proof


def verify_merkle_proof(
    leaf_hash: str,
    proof: list[MerkleProofStep] | list[dict[str, str]],
    expected_root: str,
) -> bool:
    """Verify that `leaf_hash` alongside `proof` hashes directly to `expected_root`.

    Returns True if valid, False if tampered or invalid.
    """
    current = leaf_hash

    for step in proof:
        if isinstance(step, dict):
            sibling = step["sibling_hash"]
            pos = step["position"]
        else:
            sibling = step.sibling_hash
            pos = step.position

        if pos == "right":
            current = hash_pair(current, sibling)
        elif pos == "left":
            current = hash_pair(sibling, current)
        else:
            return False

    return current.lower() == expected_root.lower()

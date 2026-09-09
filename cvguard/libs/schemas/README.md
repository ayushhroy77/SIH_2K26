# CVGuard Shared Schemas (`cvguard-schemas`)

Canonical, versioned Pydantic v2 domain schemas for CVGuard.
Provides strongly typed models for `Finding`, `SignedFinding`, `Report`, `CoverageStatement`, and reproducibility manifests across all CVGuard planes.

---

## Changelog & Version History

### Version 0.2.0 (Phase 1: Governance Spine) — BREAKING CHANGES
1. **Locked `AssetType` Enum on `Finding`:**
   - Changed `Finding.asset_type` from a free-form string to the locked enum `AssetType` (`sample`, `source`, `model`, `inference_record`, `batch`).
   - Free-form asset type strings (such as `"dataset_image"` or `"model_weights"`) will fail Pydantic validation and must be migrated to their canonical enum equivalents (e.g. `AssetType.SAMPLE`, `AssetType.MODEL`).
2. **Finding Immutability & Signature Separation:**
   - Removed the optional `signature` field directly from `Finding` to guarantee that `Finding` remains a purely content-bound, frozen model (`frozen=True`, `extra="forbid"`).
   - Introduced `SignedFinding` as the canonical envelope containing `finding: Finding`, `entry_hash: str`, `prev_hash: str`, `signature: str`, and `ledger_id: int | None`. This avoids mutating or re-constructing domain findings post-signing.

### Version 0.1.0 (Phase 0: Scaffold)
- Initial release containing base `Finding`, `Report`, `Severity`, `Disposition`, and `CoverageStatement` schemas.

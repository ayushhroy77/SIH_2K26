"""Cryptographic signing and verification service for CVGuard Governance.

Provides an abstract Signer base class and a concrete LocalFileSigner implementation
using Ed25519 (Edwards-curve Digital Signature Algorithm).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger("cvguard.governance.signer")


class Signer(ABC):
    """Abstract cryptographic signer base class.

    Architectural decoupling: Allows later phases to cleanly swap in hardware
    security modules (HSMs), HashiCorp Vault Transit engine, or Confidential Computing
    TEE-backed signers (e.g. AWS Nitro Enclaves, Intel SGX) without altering any
    calling governance or ledger logic.
    """

    @abstractmethod
    def sign(self, payload: bytes) -> str:
        """Sign arbitrary payload bytes and return hex-encoded signature string.

        Args:
            payload: Deterministic canonical bytes to be signed.

        Returns:
            Lowercase hex-encoded signature string (128 characters for Ed25519).
        """
        raise NotImplementedError

    @abstractmethod
    def verify(
        self,
        payload: bytes,
        signature: str,
        public_key: bytes | None = None,
    ) -> bool:
        """Verify an Ed25519 signature against payload bytes.

        Args:
            payload: The canonical bytes that were supposedly signed.
            signature: Lowercase hex-encoded signature string.
            public_key: Optional raw 32-byte Ed25519 public key. If None, uses
                        this signer's own public key.

        Returns:
            True if cryptographic signature is valid, False otherwise.
            Never raises on signature mismatch.
        """
        raise NotImplementedError

    @abstractmethod
    def get_public_key_bytes(self) -> bytes:
        """Return raw 32-byte public key representation."""
        raise NotImplementedError

    @abstractmethod
    def get_public_key_hex(self) -> str:
        """Return hex-encoded public key string (64 characters for Ed25519)."""
        raise NotImplementedError


class LocalFileSigner(Signer):
    """Ed25519 Signer implementation backed by a local private key file.

    ==============================================================================
    DEV-ONLY PLACEHOLDER NOTICE:
    Storing raw private keys on the local filesystem is strictly intended for local
    development, testing, and initial air-gapped staging. In Phase 7+ (TEE / Hardware
    Root of Trust), this file-backed signer will be replaced by a hardware enclave or
    Vault Transit-backed implementation.
    The private key is guarded with strict OS file permissions (0600) and must NEVER
    appear in logs, API responses, or error messages.
    ==============================================================================
    """

    def __init__(self, key_path: str | Path | None = None) -> None:
        """Initialize or load the Ed25519 keypair.

        If the key file does not exist, a fresh Ed25519 keypair is generated,
        stored in the target directory with restricted permissions, and loaded into memory.
        """
        resolved_path = (
            Path(key_path)
            if key_path
            else Path(os.getenv("CVGUARD_SIGNING_KEY_PATH", "secrets/governance_ed25519.pem"))
        )
        self._key_path = resolved_path.resolve()
        self._private_key: ed25519.Ed25519PrivateKey
        self._public_key: ed25519.Ed25519PublicKey

        self._load_or_generate_key()

    def _load_or_generate_key(self) -> None:
        """Load private key from disk or generate a new Ed25519 keypair."""
        if self._key_path.is_file():
            try:
                pem_data = self._key_path.read_bytes()
                loaded_key = serialization.load_pem_private_key(pem_data, password=None)
                if not isinstance(loaded_key, ed25519.Ed25519PrivateKey):
                    raise ValueError(
                        f"Expected Ed25519PrivateKey, got {type(loaded_key).__name__}"
                    )
                self._private_key = loaded_key
                self._public_key = loaded_key.public_key()
                logger.info(
                    "Loaded existing Ed25519 governance signing key from %s",
                    self._key_path,
                )
            except Exception as exc:
                # Do NOT log key contents or sensitive stack traces with key data
                logger.error(
                    "Failed to load signing key from %s: %s",
                    self._key_path,
                    type(exc).__name__,
                )
                raise RuntimeError(
                    f"Could not load governance signing key from {self._key_path}"
                ) from exc
        else:
            # Generate fresh Ed25519 keypair
            logger.warning(
                "No signing key found at %s. Generating fresh Ed25519 keypair (DEV-ONLY).",
                self._key_path,
            )
            self._private_key = ed25519.Ed25519PrivateKey.generate()
            self._public_key = self._private_key.public_key()

            # Ensure parent secrets directory exists with 0700 permissions
            parent_dir = self._key_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(parent_dir, 0o700)
            except OSError:
                pass  # Non-fatal on non-POSIX / some container mounts

            # Serialize private key to PKCS8 PEM
            pem_bytes = self._private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

            # Write key file with strict 0600 permissions
            self._key_path.write_bytes(pem_bytes)
            try:
                os.chmod(self._key_path, 0o600)
            except OSError:
                pass

            logger.info("Generated and saved new Ed25519 key to %s", self._key_path)

    def sign(self, payload: bytes) -> str:
        """Sign canonical bytes with the Ed25519 private key.

        Returns:
            Hex-encoded detached signature string (128 hex chars = 64 bytes).
        """
        signature_bytes = self._private_key.sign(payload)
        return signature_bytes.hex()

    def verify(
        self,
        payload: bytes,
        signature: str,
        public_key: bytes | None = None,
    ) -> bool:
        """Verify an Ed25519 signature.

        Returns:
            True if verification succeeds, False if signature is invalid or corrupt.
        """
        try:
            sig_bytes = bytes.fromhex(signature)
            if len(sig_bytes) != 64:
                return False

            if public_key is not None:
                if len(public_key) != 32:
                    return False
                verifier = ed25519.Ed25519PublicKey.from_public_bytes(public_key)
            else:
                verifier = self._public_key

            verifier.verify(sig_bytes, payload)
            return True
        except (InvalidSignature, ValueError):
            return False
        except Exception:
            return False

    def get_public_key_bytes(self) -> bytes:
        """Return raw 32-byte Ed25519 public key."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def get_public_key_hex(self) -> str:
        """Return hex-encoded 32-byte Ed25519 public key (64 hex characters)."""
        return self.get_public_key_bytes().hex()

    def __repr__(self) -> str:
        """Safety override: prevent accidental leakage of key path or sensitive attributes."""
        return f"<LocalFileSigner public_key={self.get_public_key_hex()[:16]}...>"

    def __str__(self) -> str:
        return f"LocalFileSigner(pubkey_prefix={self.get_public_key_hex()[:12]})"

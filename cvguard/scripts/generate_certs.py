#!/usr/bin/env python3
"""CVGuard Internal Mutual TLS (mTLS) Certificate Authority and Leaf Generator.

Phase 8 Hardening Pass:
Generates an offline root Certificate Authority (CA) and leaf certificate/key pairs
for each of the six CVGuard microservices (gateway, data-plane, model-plane,
inference-plane, drift-plane, governance).

Each leaf certificate includes:
- Subject Alternative Name (SAN) matching its docker-compose service DNS hostname
- ExtendedKeyUsage enabling both Server Authentication (id-kp-serverAuth) and
  Client Authentication (id-kp-clientAuth) for strict bidirectional mTLS
- Private keys saved with strict POSIX permissions (0600) in untracked certs directory.
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
import os
from pathlib import Path
from typing import Sequence

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

SERVICES: list[str] = [
    "gateway",
    "data-plane",
    "model-plane",
    "inference-plane",
    "drift-plane",
    "governance",
]


def generate_private_key(key_size: int = 2048) -> rsa.RSAPrivateKey:
    """Generate a standard RSA private key."""
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )


def save_private_key(key: rsa.RSAPrivateKey, path: Path) -> None:
    """Save private key in unencrypted PEM format with POSIX 0600 permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pem_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    # Write file and restrict permissions
    with open(path, "wb") as f:
        f.write(pem_bytes)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def save_certificate(cert: x509.Certificate, path: Path) -> None:
    """Save X.509 certificate in PEM format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


def create_root_ca(
    common_name: str = "CVGuard Internal Root CA",
    days_valid: int = 3650,
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Create a self-signed Root Certificate Authority."""
    ca_key = generate_private_key(key_size=4096)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CVGuard Assurance"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Internal PKI"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    cert_builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days_valid))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=1),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
    )

    ca_cert = cert_builder.sign(ca_key, hashes.SHA256())
    return ca_key, ca_cert


def issue_leaf_cert(
    service_name: str,
    ca_key: rsa.RSAPrivateKey,
    ca_cert: x509.Certificate,
    additional_dns_names: Sequence[str] | None = None,
    days_valid: int = 365,
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Issue an mTLS leaf certificate for a specific service.

    Enables both serverAuth and clientAuth to allow symmetric bidirectional TLS.
    """
    leaf_key = generate_private_key(key_size=2048)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CVGuard Assurance"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Plane Service"),
            x509.NameAttribute(NameOID.COMMON_NAME, service_name),
        ]
    )

    # Compile SANs: service name, localhost, 127.0.0.1, plus any extras
    dns_names = [service_name, "localhost"]
    if additional_dns_names:
        dns_names.extend(additional_dns_names)

    san_names: list[x509.GeneralName] = [
        x509.DNSName(dns) for dns in sorted(set(dns_names))
    ]
    san_names.append(x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")))

    now = datetime.datetime.now(datetime.timezone.utc)
    cert_builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days_valid))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [
                    ExtendedKeyUsageOID.SERVER_AUTH,
                    ExtendedKeyUsageOID.CLIENT_AUTH,
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.SubjectAlternativeName(san_names),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
    )

    leaf_cert = cert_builder.sign(ca_key, hashes.SHA256())
    return leaf_key, leaf_cert


def build_pki(output_dir: Path, rogue: bool = False) -> None:
    """Generate complete PKI tree in target directory."""
    prefix = "rogue_" if rogue else ""
    ca_name = f"CVGuard {'Rogue ' if rogue else ''}Root CA"

    ca_key, ca_cert = create_root_ca(common_name=ca_name)

    ca_cert_path = output_dir / f"{prefix}ca.crt"
    ca_key_path = output_dir / f"{prefix}ca.key"

    save_certificate(ca_cert, ca_cert_path)
    save_private_key(ca_key, ca_key_path)

    for service in SERVICES:
        srv_dir = output_dir / f"{prefix}{service}"
        leaf_key, leaf_cert = issue_leaf_cert(
            service_name=service,
            ca_key=ca_key,
            ca_cert=ca_cert,
            additional_dns_names=[f"cvguard-{service}"],
        )
        save_certificate(leaf_cert, srv_dir / f"{service}.crt")
        save_private_key(leaf_key, srv_dir / f"{service}.key")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CVGuard internal PKI certificates for mTLS."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infra/certs"),
        help="Directory where CA and service certificates are written.",
    )
    parser.add_argument(
        "--rogue",
        action="store_true",
        help="Generate a separate rogue CA and rogue service certificates for security testing.",
    )
    args = parser.parse_args()

    build_pki(output_dir=args.output_dir, rogue=args.rogue)


if __name__ == "__main__":
    main()

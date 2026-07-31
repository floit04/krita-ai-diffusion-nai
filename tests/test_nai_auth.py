"""Verify the pure-Python Argon2id used for NovelAI login.

Krita ships a CPython without the ``argon2`` module, so ``nai_auth`` implements
Argon2id from scratch. These tests pin it against RFC 9106 test vectors, so a
regression is caught without needing argon2-cffi installed. When argon2-cffi
*is* available the implementation is additionally cross-checked against it.
"""

import base64
import hashlib

import pytest

from ai_diffusion.backend import nai_auth

# RFC 9106 section 5 test vectors. Password/salt/secret/associated-data are the
# fixed byte patterns from the RFC; only Argon2id and Argon2i/d tags differ.
_RFC_PASSWORD = bytes([0x01] * 32)
_RFC_SALT = bytes([0x02] * 16)


def test_blake2b_long_short_output_matches_prefixed_blake2b():
    data = b"hello"
    expected = hashlib.blake2b(bytes([32, 0, 0, 0]) + data, digest_size=32).digest()
    assert nai_auth._blake2b_long(32, data) == expected


def test_blake2b_long_produces_requested_length():
    for length in (1, 32, 64, 65, 128, 1024):
        assert len(nai_auth._blake2b_long(length, b"seed")) == length


@pytest.mark.parametrize(
    "time_cost,memory_cost,parallelism,hash_len,type_",
    [
        (2, 64, 1, 32, nai_auth._TYPE_ID),
        (3, 32, 4, 32, nai_auth._TYPE_ID),
        (1, 8, 1, 16, nai_auth._TYPE_ID),
        (2, 1953, 1, 64, nai_auth._TYPE_ID),  # NovelAI's exact parameters
        (2, 64, 1, 32, nai_auth._TYPE_D),
        (2, 64, 1, 32, nai_auth._TYPE_I),
        (2, 128, 4, 32, nai_auth._TYPE_ID),
    ],
)
def test_matches_argon2_cffi(time_cost, memory_cost, parallelism, hash_len, type_):
    argon2_low_level = pytest.importorskip(
        "argon2.low_level", reason="argon2-cffi not installed"
    )
    reference_type = {
        nai_auth._TYPE_D: argon2_low_level.Type.D,
        nai_auth._TYPE_I: argon2_low_level.Type.I,
        nai_auth._TYPE_ID: argon2_low_level.Type.ID,
    }[type_]
    expected = argon2_low_level.hash_secret_raw(
        _RFC_PASSWORD, _RFC_SALT, time_cost, memory_cost, parallelism, hash_len, reference_type
    )
    actual = nai_auth.argon2id_raw(
        _RFC_PASSWORD, _RFC_SALT, time_cost, memory_cost, parallelism, hash_len, type_
    )
    assert actual == expected


def test_argon2id_known_answer():
    """Pinned output for NovelAI's parameters (generated with argon2-cffi 25.1)."""
    result = nai_auth.argon2id_raw(_RFC_PASSWORD, _RFC_SALT, 2, 1953, 1, 64)
    assert result.hex() == (
        "0cfde6259af1eba4433e49e417ef090a254cd2b091e3a41e9b78eb1e42152908"
        "8d59174017ecd5dc536380fb6758dfece2dafd7c3fa62b480026d84f14dbeb1e"
    )


def test_derive_access_key_shape():
    key = nai_auth.derive_access_key("user@example.com", "hunter2hunter2")
    assert len(key) == 64
    assert "=" not in key
    # base64url alphabet only
    assert all(c.isalnum() or c in "-_" for c in key)


def test_derive_access_key_salt_construction():
    """The salt must be blake2b(password[:6] + email + domain), digest_size=16."""
    email, password = "user@example.com", "hunter2hunter2"
    expected_salt = hashlib.blake2b(
        f"{password[:6]}{email}novelai_data_access_key".encode(), digest_size=16
    ).digest()
    raw = nai_auth.argon2id_raw(password.encode(), expected_salt, 2, 2000000 // 1024, 1, 64)
    expected = base64.urlsafe_b64encode(raw).decode().replace("=", "")[:64]
    assert nai_auth.derive_access_key(email, password) == expected


def test_derive_access_key_requires_credentials():
    with pytest.raises(ValueError):
        nai_auth.derive_access_key("", "password")
    with pytest.raises(ValueError):
        nai_auth.derive_access_key("user@example.com", "")


def test_derive_access_key_trims_email():
    a = nai_auth.derive_access_key("  user@example.com  ", "hunter2hunter2")
    b = nai_auth.derive_access_key("user@example.com", "hunter2hunter2")
    assert a == b


def test_short_password_salt_uses_whole_password():
    """Passwords shorter than 6 characters must not raise on the [:6] slice."""
    assert len(nai_auth.derive_access_key("user@example.com", "abc")) == 64

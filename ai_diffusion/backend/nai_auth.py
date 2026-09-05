"""NovelAI e-mail + password login.

NovelAI never sends the password to the server. The client derives an *access
key* from (e-mail, password) with Argon2id and posts only that key to
``/user/login``, which answers with a JWT bearer token. Algorithm, verbatim
from the launcher's ``NAICryptoService`` (which matches novelai.net and the
``novelai-api`` reference implementation)::

    pre_salt   = password[:6] + email + "novelai_data_access_key"
    salt       = blake2b(pre_salt, digest_size=16)
    raw        = argon2id(password, salt, t=2, m=2000000//1024, p=1, len=64)
    access_key = base64url(raw).replace("=", "")[:64]

Krita ships a bare CPython with no ``argon2`` module and no way to pip-install
one, so Argon2id is implemented here in pure Python (``hashlib.blake2b`` is
available, which is the only primitive needed). It is verified byte-for-byte
against argon2-cffi in ``tests/test_nai_auth.py``.

At NovelAI's parameters (m=1953 KiB, t=2, p=1) this takes a few seconds — much
cheaper than the usual 64 MiB defaults, but far too slow for the UI thread.
Call ``derive_access_key`` from a worker thread.
"""

from __future__ import annotations

import base64
import hashlib
import struct

_MASK64 = 0xFFFFFFFFFFFFFFFF
_MASK32 = 0xFFFFFFFF

_BLOCK_WORDS = 128  # 1024-byte block = 128 little-endian uint64
_SYNC_POINTS = 4
_ADDRESSES_IN_BLOCK = 128
_VERSION = 0x13
_TYPE_D = 0
_TYPE_I = 1
_TYPE_ID = 2

# Blake2 round applied to 8 rows, then to 8 columns of the 128-word block.
_ROW_INDICES = tuple(tuple(range(16 * r, 16 * r + 16)) for r in range(8))
_COL_INDICES = tuple(
    tuple(idx for group in range(8) for idx in (2 * c + 16 * group, 2 * c + 16 * group + 1))
    for c in range(8)
)


def _g(a: int, b: int, c: int, d: int):
    """The Argon2 G function (Blake2b round with the fBlaMka mixing)."""
    a = (a + b + 2 * (a & _MASK32) * (b & _MASK32)) & _MASK64
    d ^= a
    d = ((d >> 32) | (d << 32)) & _MASK64
    c = (c + d + 2 * (c & _MASK32) * (d & _MASK32)) & _MASK64
    b ^= c
    b = ((b >> 24) | (b << 40)) & _MASK64
    a = (a + b + 2 * (a & _MASK32) * (b & _MASK32)) & _MASK64
    d ^= a
    d = ((d >> 16) | (d << 48)) & _MASK64
    c = (c + d + 2 * (c & _MASK32) * (d & _MASK32)) & _MASK64
    b ^= c
    b = ((b >> 63) | (b << 1)) & _MASK64
    return a, b, c, d


def _permute(v: list[int], idx: tuple[int, ...]):
    i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15 = idx
    v[i0], v[i4], v[i8], v[i12] = _g(v[i0], v[i4], v[i8], v[i12])
    v[i1], v[i5], v[i9], v[i13] = _g(v[i1], v[i5], v[i9], v[i13])
    v[i2], v[i6], v[i10], v[i14] = _g(v[i2], v[i6], v[i10], v[i14])
    v[i3], v[i7], v[i11], v[i15] = _g(v[i3], v[i7], v[i11], v[i15])
    v[i0], v[i5], v[i10], v[i15] = _g(v[i0], v[i5], v[i10], v[i15])
    v[i1], v[i6], v[i11], v[i12] = _g(v[i1], v[i6], v[i11], v[i12])
    v[i2], v[i7], v[i8], v[i13] = _g(v[i2], v[i7], v[i8], v[i13])
    v[i3], v[i4], v[i9], v[i14] = _g(v[i3], v[i4], v[i9], v[i14])


def _fill_block(prev: list[int], ref: list[int], nxt: list[int], with_xor: bool):
    """Compression G(prev, ref) written into ``nxt`` (in place)."""
    block = [a ^ b for a, b in zip(prev, ref)]
    if with_xor:
        tmp = [a ^ b for a, b in zip(block, nxt)]
    else:
        tmp = list(block)
    for idx in _ROW_INDICES:
        _permute(block, idx)
    for idx in _COL_INDICES:
        _permute(block, idx)
    nxt[:] = [a ^ b for a, b in zip(tmp, block)]


def _blake2b_long(out_len: int, data: bytes) -> bytes:
    """Argon2's variable-length hash H' (RFC 9106 section 3.2)."""
    prefix = struct.pack("<I", out_len)
    if out_len <= 64:
        return hashlib.blake2b(prefix + data, digest_size=out_len).digest()
    buf = hashlib.blake2b(prefix + data, digest_size=64).digest()
    out = bytearray(buf[:32])
    to_produce = out_len - 32
    while to_produce > 64:
        buf = hashlib.blake2b(buf, digest_size=64).digest()
        out += buf[:32]
        to_produce -= 32
    out += hashlib.blake2b(buf, digest_size=to_produce).digest()
    return bytes(out)


def _initial_hash(
    password: bytes,
    salt: bytes,
    time_cost: int,
    memory_cost: int,
    lanes: int,
    tag_len: int,
    type_: int,
) -> bytes:
    h = hashlib.blake2b(digest_size=64)
    for value in (lanes, tag_len, memory_cost, time_cost, _VERSION, type_):
        h.update(struct.pack("<I", value))
    for blob in (password, salt, b"", b""):  # secret and associated data are unused
        h.update(struct.pack("<I", len(blob)))
        h.update(blob)
    return h.digest()


def _bytes_to_block(data: bytes) -> list[int]:
    return list(struct.unpack("<128Q", data))


def _block_to_bytes(block: list[int]) -> bytes:
    return struct.pack("<128Q", *block)


def _index_alpha(
    pass_: int,
    slice_: int,
    index: int,
    pseudo_rand: int,
    same_lane: bool,
    lane_length: int,
    segment_length: int,
) -> int:
    if pass_ == 0:
        if slice_ == 0:
            area = index - 1
        elif same_lane:
            area = slice_ * segment_length + index - 1
        else:
            area = slice_ * segment_length + (-1 if index == 0 else 0)
    elif same_lane:
        area = lane_length - segment_length + index - 1
    else:
        area = lane_length - segment_length + (-1 if index == 0 else 0)

    relative = (pseudo_rand * pseudo_rand) >> 32
    relative = area - 1 - ((area * relative) >> 32)
    start = 0
    if pass_ != 0:
        start = 0 if slice_ == _SYNC_POINTS - 1 else (slice_ + 1) * segment_length
    return (start + relative) % lane_length


def argon2id_raw(
    password: bytes,
    salt: bytes,
    time_cost: int,
    memory_cost: int,
    parallelism: int,
    hash_len: int,
    type_: int = _TYPE_ID,
) -> bytes:
    """Argon2 (version 0x13) key derivation, pure Python.

    ``memory_cost`` is in KiB, matching argon2-cffi's ``hash_secret_raw``.
    """
    if parallelism < 1:
        raise ValueError("parallelism must be >= 1")
    if memory_cost < 8 * parallelism:
        raise ValueError("memory_cost must be at least 8 * parallelism")
    if time_cost < 1:
        raise ValueError("time_cost must be >= 1")

    lanes = parallelism
    segment_length = memory_cost // (lanes * _SYNC_POINTS)
    lane_length = segment_length * _SYNC_POINTS
    memory_blocks = lane_length * lanes

    # H0 uses the *requested* memory cost, not the value rounded down above.
    h0 = _initial_hash(password, salt, time_cost, memory_cost, lanes, hash_len, type_)

    memory: list[list[int]] = [[] for _ in range(memory_blocks)]
    for lane in range(lanes):
        for column in (0, 1):
            seed = h0 + struct.pack("<I", column) + struct.pack("<I", lane)
            memory[lane * lane_length + column] = _bytes_to_block(_blake2b_long(1024, seed))

    zero_block = [0] * _BLOCK_WORDS
    for pass_ in range(time_cost):
        for slice_ in range(_SYNC_POINTS):
            for lane in range(lanes):
                data_independent = type_ == _TYPE_I or (
                    type_ == _TYPE_ID and pass_ == 0 and slice_ < 2
                )
                address_block: list[int] = [0] * _BLOCK_WORDS
                input_block: list[int] = [0] * _BLOCK_WORDS
                if data_independent:
                    input_block[0] = pass_
                    input_block[1] = lane
                    input_block[2] = slice_
                    input_block[3] = memory_blocks
                    input_block[4] = time_cost
                    input_block[5] = type_

                starting_index = 0
                if pass_ == 0 and slice_ == 0:
                    starting_index = 2  # first two blocks of the lane already exist
                    if data_independent:
                        input_block[6] += 1
                        _fill_block(zero_block, input_block, address_block, False)
                        _fill_block(zero_block, address_block, address_block, False)

                curr = lane * lane_length + slice_ * segment_length + starting_index
                if curr % lane_length == 0:
                    prev = curr + lane_length - 1
                else:
                    prev = curr - 1

                for index in range(starting_index, segment_length):
                    if curr % lane_length == 1:
                        prev = curr - 1

                    if data_independent:
                        if index % _ADDRESSES_IN_BLOCK == 0:
                            input_block[6] += 1
                            _fill_block(zero_block, input_block, address_block, False)
                            _fill_block(zero_block, address_block, address_block, False)
                        pseudo_rand = address_block[index % _ADDRESSES_IN_BLOCK]
                    else:
                        pseudo_rand = memory[prev][0]

                    if pass_ == 0 and slice_ == 0:
                        ref_lane = lane
                    else:
                        ref_lane = (pseudo_rand >> 32) % lanes
                    ref_index = _index_alpha(
                        pass_,
                        slice_,
                        index,
                        pseudo_rand & _MASK32,
                        ref_lane == lane,
                        lane_length,
                        segment_length,
                    )

                    with_xor = pass_ != 0
                    _fill_block(
                        memory[prev],
                        memory[ref_lane * lane_length + ref_index],
                        memory[curr],
                        with_xor,
                    )
                    curr += 1
                    prev += 1

    final = list(memory[lane_length - 1])
    for lane in range(1, lanes):
        other = memory[lane * lane_length + lane_length - 1]
        final = [a ^ b for a, b in zip(final, other)]
    return _blake2b_long(hash_len, _block_to_bytes(final))


# ---------------------------------------------------------------------------
# NovelAI access key
# ---------------------------------------------------------------------------

_ACCESS_KEY_DOMAIN = "novelai_data_access_key"
_ARGON_TIME_COST = 2
_ARGON_MEMORY_COST = 2000000 // 1024  # 1953 KiB — NovelAI's exact parameter
_ARGON_PARALLELISM = 1


def derive_access_key(email: str, password: str) -> str:
    """Derive the NovelAI login access key from e-mail and password.

    Takes several seconds — do not call this on the UI thread.
    """
    email = email.strip()
    if not email or not password:
        raise ValueError("e-mail and password are required")

    pre_salt = f"{password[:6]}{email}{_ACCESS_KEY_DOMAIN}"
    salt = hashlib.blake2b(pre_salt.encode("utf-8"), digest_size=16).digest()
    raw = argon2id_raw(
        password.encode("utf-8"),
        salt,
        _ARGON_TIME_COST,
        _ARGON_MEMORY_COST,
        _ARGON_PARALLELISM,
        64,
    )
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").replace("=", "")
    return encoded[:64]

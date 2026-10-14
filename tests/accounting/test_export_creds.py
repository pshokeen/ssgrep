"""Ingest-credential references: owner-only files, send-time env, separation from local secrets."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ssgrep.accounting import datadir
from ssgrep.accounting.export_creds import (
    Credential,
    CredentialError,
    check_ref,
    resolve,
)

CANARY = "ingest-CANARY-9c41e7"
PROFILE = "acme"


@pytest.fixture
def secrets_dir(tmp_path: Path) -> Path:
    path = tmp_path / "operator-secrets"
    path.mkdir(mode=0o700)
    return path


def _secret(
    directory: Path, content: str | bytes = CANARY, mode: int = 0o600, name="ingest"
) -> Path:
    path = directory / name
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    path.chmod(mode)
    return path


def _value(credential: Credential | None) -> str:
    assert credential is not None
    return credential.value


def _profile_dir() -> Path:
    path = datadir.data_root() / "accounting" / PROFILE
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def _error(ref: str, *, resolving: bool = False, environ=None) -> CredentialError:
    with pytest.raises(CredentialError) as caught:
        if resolving:
            resolve(PROFILE, ref, environ=environ)
        else:
            check_ref(PROFILE, ref)
    return caught.value


# ------------------------------------------------------------------ files: accepted


@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_owner_only_file_is_accepted_and_resolves_to_a_bearer_header(secrets_dir, mode):
    path = _secret(secrets_dir, CANARY + "\n", mode)
    check_ref(PROFILE, f"file:{path}")
    credential = resolve(PROFILE, f"file:{path}")
    assert credential == Credential("Authorization", f"Bearer {CANARY}")


def test_tilde_is_expanded(secrets_dir, monkeypatch):
    monkeypatch.setenv("HOME", str(secrets_dir))
    _secret(secrets_dir, CANARY, name="tok")
    credential = resolve(PROFILE, "file:~/tok")
    assert credential is not None and credential.value == f"Bearer {CANARY}"


def test_no_reference_means_no_auth():
    assert resolve(PROFILE, None) is None
    check_ref(PROFILE, None)


# ------------------------------------------------------------------ files: refused


@pytest.mark.parametrize("mode", [0o644, 0o660, 0o640, 0o666, 0o604, 0o607])
def test_group_or_other_accessible_files_are_refused_with_corrective_action(secrets_dir, mode):
    path = _secret(secrets_dir, CANARY, mode)
    for resolving in (False, True):
        error = _error(f"file:{path}", resolving=resolving)
        assert error.condition == "credential_unsafe_file"
        assert f"chmod 600 {path}" in error.action
        assert CANARY not in str(error) + error.action + repr(error)


def test_symlink_final_component_is_refused_even_to_a_good_file(secrets_dir):
    real = _secret(secrets_dir, CANARY)
    link = secrets_dir / "link"
    link.symlink_to(real)
    for resolving in (False, True):
        assert _error(f"file:{link}", resolving=resolving).condition == "credential_unsafe_file"


def test_missing_file_is_reported_as_credential_missing(secrets_dir):
    error = _error(f"file:{secrets_dir / 'nope'}")
    assert error.condition == "credential_missing"
    assert str(secrets_dir / "nope") in error.action


def test_directory_and_fifo_are_not_regular_files(secrets_dir):
    assert _error(f"file:{secrets_dir}").condition == "credential_unsafe_file"
    fifo = secrets_dir / "fifo"
    os.mkfifo(fifo, 0o600)
    assert _error(f"file:{fifo}").condition == "credential_unsafe_file"


@pytest.mark.parametrize(
    ("content", "condition"),
    [
        (b"", "credential_malformed"),
        (b"   \n\n", "credential_malformed"),
        (b"x" * 4097, "credential_malformed"),
        (f"{CANARY}\nsecond-line\n".encode(), "credential_malformed"),
        (f"{CANARY}\r\n{CANARY}".encode(), "credential_malformed"),
        (f"a b{CANARY}".encode(), "credential_malformed"),
        (f"{CANARY}\x00x".encode(), "credential_malformed"),
        (f"{CANARY}\x1b[0m".encode(), "credential_malformed"),
        (f"tök{CANARY}".encode(), "credential_malformed"),
        (b"\xff\xfe" + CANARY.encode(), "credential_malformed"),
    ],
)
def test_malformed_content_is_refused_without_echoing_it(secrets_dir, content, condition):
    path = _secret(secrets_dir, content)
    for resolving in (False, True):
        error = _error(f"file:{path}", resolving=resolving)
        assert error.condition == condition
        assert CANARY not in str(error) + error.action


def test_exactly_4096_bytes_is_accepted(secrets_dir):
    path = _secret(secrets_dir, "k" * 4096)
    assert _value(resolve(PROFILE, f"file:{path}")) == "Bearer " + "k" * 4096


@pytest.mark.parametrize(
    "ref",
    ["", "http://x", "env", "env:", "env:lower", "env:1ABC", "env:" + "A" * 65, "file:",
     "file:relative/path", "file:./x", "vault:abc", "env:A B", "file:/x\x00y"],
)  # fmt: skip
def test_invalid_reference_shapes_are_refused(ref):
    assert _error(ref).condition == "credential_ref_invalid"
    assert _error(ref, resolving=True).condition == "credential_ref_invalid"


def test_invalid_profile_is_refused():
    with pytest.raises(CredentialError) as caught:
        check_ref("../evil", "env:TOKEN")
    assert caught.value.condition == "profile_invalid"


# ------------------------------------------------------------------ env: send time only


def test_env_is_read_only_at_resolve_time_and_never_cached(monkeypatch):
    monkeypatch.delenv("ACME_INGEST", raising=False)
    check_ref(PROFILE, "env:ACME_INGEST")  # unset at config time is fine: no read happens
    monkeypatch.setenv("ACME_INGEST", "first-value")
    assert _value(resolve(PROFILE, "env:ACME_INGEST")) == "Bearer first-value"
    check_ref(PROFILE, "env:ACME_INGEST")
    monkeypatch.setenv("ACME_INGEST", "second-value")
    assert _value(resolve(PROFILE, "env:ACME_INGEST")) == "Bearer second-value"


def test_check_ref_does_not_read_the_environment(monkeypatch):
    class Tripwire(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("environment was read at config time")

        __getitem__ = __contains__ = get

    check_ref(PROFILE, "env:ACME_INGEST", environ=Tripwire())
    with pytest.raises(AssertionError):
        resolve(PROFILE, "env:ACME_INGEST", environ=Tripwire())  # positive twin: resolve reads


def test_unset_or_blank_env_is_credential_missing(monkeypatch):
    monkeypatch.delenv("ACME_INGEST", raising=False)
    error = _error("env:ACME_INGEST", resolving=True)
    assert error.condition == "credential_missing" and "ACME_INGEST" in error.action
    for blank in ("", "  \n"):
        assert _error(
            "env:ACME_INGEST", resolving=True, environ={"ACME_INGEST": blank}
        ).condition == ("credential_missing")


@pytest.mark.parametrize("value", ["a\r\nX-Evil: 1", "a\nb", "a\x00b", "a b", "a\tb"])
def test_env_values_with_control_characters_are_refused(value):
    error = _error("env:ACME_INGEST", resolving=True, environ={"ACME_INGEST": value})
    assert error.condition == "credential_malformed"
    assert "X-Evil" not in str(error)


def test_env_value_is_stripped(monkeypatch):
    credential = resolve(PROFILE, "env:ACME_INGEST", environ={"ACME_INGEST": f" {CANARY}\n"})
    assert _value(credential) == f"Bearer {CANARY}"


# ------------------------------------------------------------------ separation from local secrets


def test_local_endpoint_token_file_is_never_an_ingest_credential(secrets_dir, tmp_path):
    token = "ab" * 32
    profile_dir = _profile_dir()
    _secret(profile_dir, token + "\n", name="endpoint.token")
    # a hard link outside accounting storage is the same inode
    linked = secrets_dir / "linked"
    os.link(profile_dir / "endpoint.token", linked)
    # a byte-for-byte copy elsewhere
    copy = _secret(secrets_dir, token + "\n", name="copy")
    for path in (linked, copy):
        for resolving in (False, True):
            assert _error(f"file:{path}", resolving=resolving).condition == (
                "credential_reuses_local_secret"
            )
    # positive twin: a different token in the same place is accepted
    other = _secret(secrets_dir, "cd" * 32, name="other")
    check_ref(PROFILE, f"file:{other}")


@pytest.mark.parametrize("name", ["fingerprint.key", "export/pseudonym.key"])
def test_fingerprint_and_pseudonym_key_copies_are_refused(secrets_dir, name):
    profile_dir = _profile_dir()
    target = profile_dir / name
    target.parent.mkdir(exist_ok=True, mode=0o700)
    key = bytes(range(33, 65))  # printable 32-byte key: a copy is a plausible operator mistake
    _secret(target.parent, key, name=target.name)
    copy = _secret(secrets_dir, key, name="copy")
    assert _error(f"file:{copy}").condition == "credential_reuses_local_secret"
    assert _error(f"file:{copy}", resolving=True).condition == "credential_reuses_local_secret"


def test_env_copy_of_a_local_secret_is_refused_at_send_time(secrets_dir):
    token = "ef" * 32
    _secret(_profile_dir(), token, name="endpoint.token")
    error = _error("env:COPY", resolving=True, environ={"COPY": token})
    assert error.condition == "credential_reuses_local_secret"
    assert token not in str(error) + error.action
    assert resolve(PROFILE, "env:COPY", environ={"COPY": token + "x"}) is not None


@pytest.mark.parametrize("name", ["fingerprint.key", "export/pseudonym.key"])
@pytest.mark.parametrize(
    "key",
    [b"\xa5" * 32, *(bytes([edge]) + b"\xa5" * 30 + bytes([edge]) for edge in b" \t\n\r\v\f")],
)
def test_binary_key_copy_hex_encoded_in_env_is_also_refused(secrets_dir, name, key):
    target = _profile_dir() / name
    target.parent.mkdir(exist_ok=True, mode=0o700)
    _secret(target.parent, key, name=target.name)
    copied = key.hex()
    error = _error("env:COPY", resolving=True, environ={"COPY": copied})
    assert error.condition == "credential_reuses_local_secret"
    assert copied not in str(error) + error.action
    file_copy = _secret(secrets_dir, copied, name="hex-copy")
    for resolving in (False, True):
        assert _error(f"file:{file_copy}", resolving=resolving).condition == (
            "credential_reuses_local_secret"
        )
    different = (b"\xb6" * 32).hex()
    assert (
        _value(resolve(PROFILE, "env:COPY", environ={"COPY": different})) == "Bearer " + different
    )


def test_other_profiles_secrets_do_not_block_this_profile(secrets_dir):
    other = datadir.data_root() / "accounting" / "other"
    other.mkdir(parents=True, mode=0o700)
    _secret(other, CANARY, name="endpoint.token")
    path = _secret(secrets_dir, CANARY)
    check_ref(PROFILE, f"file:{path}")


def test_credentials_inside_accounting_storage_are_refused_outright():
    inside = _secret(_profile_dir(), CANARY, name="ingest")
    error = _error(f"file:{inside}")
    assert error.condition == "credential_in_accounting_storage"
    assert _error(f"file:{inside}", resolving=True).condition == "credential_in_accounting_storage"
    # via a dotted path and via a symlinked parent directory
    sneaky = f"{inside.parent}/../{PROFILE}/ingest"
    assert _error(f"file:{sneaky}").condition == "credential_in_accounting_storage"


def test_symlinked_parent_into_accounting_storage_is_refused(secrets_dir):
    inside = _secret(_profile_dir(), CANARY, name="ingest")
    alias = secrets_dir / "alias"
    alias.symlink_to(inside.parent, target_is_directory=True)
    assert _error(f"file:{alias / 'ingest'}").condition == "credential_in_accounting_storage"


# ------------------------------------------------------------------ secrecy


def test_credential_repr_and_str_redact_the_value():
    credential = resolve(PROFILE, "env:X", environ={"X": CANARY})
    assert credential is not None
    for text in (repr(credential), str(credential), f"{credential}", f"{credential!r}"):
        assert CANARY not in text and "Bearer" not in text
    assert credential.header_name in repr(credential)
    assert credential.value == f"Bearer {CANARY}"  # positive twin: the value is really there


def test_errors_never_contain_the_secret_for_any_refusal(secrets_dir):
    cases = [
        _secret(secrets_dir, CANARY, 0o644, name="a"),
        _secret(secrets_dir, f"{CANARY}\n{CANARY}", 0o600, name="b"),
        _secret(secrets_dir, CANARY.encode() + b"\x00", 0o600, name="c"),
    ]
    for path in cases:
        error = _error(f"file:{path}", resolving=True)
        blob = str(error) + error.action + error.condition + repr(error) + repr(error.args)
        assert CANARY not in blob

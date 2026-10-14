"""Declared-schema decoding rejects noncanonical data without importing data-selected types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from ssgrep.accounting import plain_codec


@dataclass
class Nested:
    rows: list[dict[str, int]]
    derived: int = field(default=17, init=False)


@dataclass
class Checked:
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise TypeError("count must not be negative")


def test_lists_decode_nested_integer_fields_and_preserve_noninit_defaults() -> None:
    decoded = plain_codec.from_plain(Nested, {"rows": [{"count": 3}]})
    assert decoded == Nested([{"count": 3}]) and decoded.derived == 17
    with pytest.raises(ValueError, match=r"rows\.0\.count: expected int"):
        plain_codec.from_plain(Nested, {"rows": [{"count": True}]})
    with pytest.raises(ValueError, match="unknown field.*derived"):
        plain_codec.from_plain(Nested, {"rows": [], "derived": 99})
    assert plain_codec.to_plain(decoded) == {"rows": [{"count": 3}], "derived": 17}


@pytest.mark.parametrize("declared", [None, type(None)])
def test_null_schema_accepts_only_null(declared: Any) -> None:
    assert plain_codec.from_plain(declared, None) is None
    with pytest.raises(ValueError, match="expected null"):
        plain_codec.from_plain(declared, False)


def test_union_attempts_each_declared_alternative_without_integer_bool_coercion() -> None:
    declared = type(None) | int | str
    assert plain_codec.from_plain(declared, "seven") == "seven"
    assert plain_codec.from_plain(declared, 7) == 7
    assert plain_codec.from_plain(declared, None) is None
    with pytest.raises(ValueError, match="declared union members"):
        plain_codec.from_plain(declared, True)


@pytest.mark.parametrize("declared", [dict[str, int], Mapping[str, int]])
def test_mapping_schema_requires_objects_and_string_keys(declared: Any) -> None:
    assert plain_codec.from_plain(declared, {"answer": 42}) == {"answer": 42}
    with pytest.raises(ValueError, match="expected object"):
        plain_codec.from_plain(declared, [["answer", 42]])
    with pytest.raises(ValueError, match="string key"):
        plain_codec.from_plain(declared, {1: 42})


def test_list_schema_rejects_python_tuple_even_when_elements_fit() -> None:
    assert plain_codec.from_plain(list[int], [1, 2]) == [1, 2]
    with pytest.raises(ValueError, match="expected list"):
        plain_codec.from_plain(list[int], (1, 2))


@pytest.mark.parametrize("declared", [float, bytes, set[int]])
def test_noncanonical_declared_types_are_not_coerced(declared: Any) -> None:
    with pytest.raises(ValueError, match="floats are not allowed|unsupported declared type"):
        plain_codec.from_plain(declared, 1)


@pytest.mark.parametrize("declared", [Any, object])
@pytest.mark.parametrize("data", [{1: "value"}, {"nested": [b"binary"]}])
def test_untyped_fields_still_reject_non_json_data(declared: Any, data: Any) -> None:
    with pytest.raises(ValueError, match="string key|nested.0: expected plain JSON value"):
        plain_codec.from_plain(declared, data)
    assert plain_codec.from_plain(declared, {"nested": [None, True, 7, "value"]}) == {
        "nested": [None, True, 7, "value"]
    }


def test_constructor_validation_is_a_decode_error_with_its_original_cause() -> None:
    assert plain_codec.from_plain(Checked, {"count": 2}) == Checked(2)
    with pytest.raises(ValueError, match="Checked rejected its fields") as caught:
        plain_codec.from_plain(Checked, {"count": -1})
    assert isinstance(caught.value.__cause__, TypeError)
    assert str(caught.value.__cause__) == "count must not be negative"


@pytest.mark.parametrize("value", [b"bytes", {1, 2}, complex(1, 2)])
def test_encoder_refuses_values_without_a_canonical_json_representation(value: Any) -> None:
    with pytest.raises(TypeError, match="cannot encode"):
        plain_codec.to_plain({"payload": value})
    assert plain_codec.to_plain({"payload": (1, None, "text")}) == {"payload": [1, None, "text"]}

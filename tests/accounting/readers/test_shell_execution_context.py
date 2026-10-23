"""Shell evidence is parsed, not executed or multiplied into journal runs."""

import shlex

import pytest

from ssgrep.accounting.ledger.contracts import InvocationClass
from ssgrep.accounting.ledger.reduce_join import shell_shape
from ssgrep.accounting.readers.classify import classify_call
from ssgrep.accounting.readers.shell_scan import MAX_DEPTH, scan_shell


def classify(command):
    return classify_call("omp", "bash", {"command": command})


@pytest.mark.parametrize(
    "prefix",
    [
        "timeout 2",
        "timeout -k 1s 2s",
        "timeout --kill-after 1s 2m",
        "timeout --kill-after=1s -- 2s",
        "timeout 2s --",
        "stdbuf -oL -eL -i0",
        "stdbuf --output=L --error=L --input=0",
        "stdbuf --output L --error L --input 0 --",
        "timeout -k1s 2s stdbuf -oL env -i uv run --quiet",
    ],
)
def test_supported_wrappers_preserve_request_arguments(prefix):
    result = classify(prefix + " ssgrep search 'alpha beta' --limit 3 --json")
    assert result.invocation_class is InvocationClass.TRANSCRIPT_ONLY
    assert result.invocation.argv == (("search", "alpha beta", "--limit", "3", "--json"),)


@pytest.mark.parametrize(
    "command",
    [
        "timeout --mystery 2 ssgrep search q",
        "timeout -k nope 2 ssgrep search q",
        "timeout ssgrep search q",
        "stdbuf --mystery ssgrep search q",
        "stdbuf -oBAD ssgrep search q",
        "env --mystery ssgrep search q",
        "uv run --mystery ssgrep search q",
        "xargs --mystery ssgrep search q",
        "bash --mystery -c 'ssgrep search q'",
        "python -c 'ssgrep search q'",
        "ruby -e 'ssgrep search q'",
        "fish -c 'ssgrep search q'",
        "sudo --mystery ssgrep search q",
        "time --mystery ssgrep search q",
        "xargs -nonsense ssgrep search q",
        "bash -n -c 'ssgrep search q'",
    ],
)
def test_unsupported_options_and_interpreters_are_unclassified(command):
    result = classify(command)
    assert result.invocation_class is InvocationClass.UNCLASSIFIED
    assert result.invocation is None


@pytest.mark.parametrize("prefix", ["sh", "bash", "bash -s", "timeout 2 stdbuf -oL bash -s --"])
@pytest.mark.parametrize(
    "redirect,body,delimiter",
    [
        ("<<'SCRIPT'", "ssgrep search 'alpha beta' --limit 3", "SCRIPT"),
        ('<<"SCRIPT"', "ssgrep search 'alpha beta' --limit 3", "SCRIPT"),
        ("<<-SCRIPT", "\tssgrep search 'alpha beta' --limit 3", "\tSCRIPT"),
        ("<<'SCRIPT-END'", "ssgrep search 'alpha beta' --limit 3", "SCRIPT-END"),
    ],
)
def test_shell_stdin_heredoc_is_script_input(prefix, redirect, body, delimiter):
    result = classify(f"{prefix} {redirect}\n{body}\n{delimiter}\n")
    assert result.invocation_class is InvocationClass.TRANSCRIPT_ONLY
    assert result.invocation.argv == (("search", "alpha beta", "--limit", "3"),)
    assert result.invocation.cardinality == "unknown"


@pytest.mark.parametrize(
    "command",
    [
        "cat <<'EOF'\nssgrep search hidden\nEOF",
        "cat <<'TEXT-END'\nssgrep search hidden\nTEXT-END",
        "echo 'bash <<EOF\nssgrep search hidden\nEOF'",
        "bash script.sh <<'EOF'\nssgrep search hidden\nEOF",
        "bash -c 'cat' <<'EOF'\nssgrep search hidden\nEOF",
        "cat <<'ONE' <<'TWO'\nssgrep search one\nONE\nssgrep show two\nTWO",
    ],
)
def test_text_heredocs_and_prompt_arguments_remain_mentions(command):
    result = classify(command)
    assert result.invocation_class is InvocationClass.MENTION
    assert result.invocation is None
    assert scan_shell(command).argvs == []


def test_text_heredoc_does_not_add_an_executable_fallback_occurrence():
    result = classify("cat <<'EOF'\nssgrep search hidden\nEOF\nssgrep show actual")
    assert result.invocation.argv == (("show", "actual"),)
    assert result.invocation.text_occurrences == 1
    assert shell_shape("cat <<'EOF'\nssgrep search hidden\nEOF\nssgrep show actual").kind == "mixed"


@pytest.mark.parametrize(
    "command",
    [
        "timeout 2 bash -c 'ssgrep search q; echo other'",
        "bash <<'EOF'\nssgrep search q\necho other\nEOF",
        "bash <<'EOF'\nfor q in one two three; do ssgrep search \"$q\"; echo other; done\nEOF",
    ],
)
def test_nested_mixed_and_loop_output_is_conservative(command):
    result = classify(command)
    assert result.invocation.text_occurrences == 1
    assert result.invocation.cardinality == "unknown"
    assert shell_shape(command).kind == "mixed"


def test_lookup_only_loop_has_pure_output_but_unknown_execution_count():
    command = "bash <<'EOF'\nfor q in one two three; do ssgrep search \"$q\"; done\nEOF"
    result = classify(command)
    assert result.invocation.text_occurrences == 1
    assert result.invocation.cardinality == "unknown"
    assert result.invocation.argv == (("search", "$q"),)
    assert shell_shape(command).kind == "pure"
    assert not scan_shell(command).has_other_commands


def test_nested_scanning_is_bounded_for_scripts_and_wrappers():
    command = "ssgrep search q"
    for _ in range(MAX_DEPTH + 2):
        command = "bash -c " + shlex.quote(command)
    assert classify(command).invocation is None
    assert classify("timeout 2 " * (MAX_DEPTH + 2) + "ssgrep search q").invocation is None


def test_parsing_never_executes_heredoc_commands(tmp_path):
    marker = tmp_path / "must-not-exist"
    result = classify(f"bash <<'EOF'\ntouch {marker}\nssgrep search q\nEOF")
    assert result.invocation.argv == (("search", "q"),)
    assert not marker.exists()


def test_tokenized_wrappers_preserve_one_unambiguous_request():
    result = classify(
        [
            "timeout",
            "-k",
            "1s",
            "2s",
            "stdbuf",
            "-oL",
            "ssgrep",
            "search",
            "alpha beta",
            "--limit",
            "3",
        ]
    )
    assert result.invocation.argv == (("search", "alpha beta", "--limit", "3"),)
    assert result.invocation.text_occurrences == 1


@pytest.mark.parametrize(
    "command",
    [
        "bash -xc 'ssgrep search q'",
        "bash -vc 'ssgrep search q'",
        "timeout --verbose 2 ssgrep search q",
        "bash -c 'ssgrep search q; python helper.py'",
        "bash -c 'ssgrep search q; ssgrep status'",
    ],
)
def test_additional_output_and_unsupported_nested_commands_are_not_pure(command):
    scan = scan_shell(command)
    assert scan.argvs == [("search", "q")]
    assert scan.has_other_commands
    assert shell_shape(command).kind == "mixed"


def test_request_text_is_not_wrapper_output_metadata():
    scan = scan_shell("timeout 2 ssgrep search '-v' --json")
    assert scan.argvs == [("search", "-v", "--json")]
    assert not scan.has_other_commands


@pytest.mark.parametrize(
    "command",
    [
        "cat <<EOF\nssgrep search q\n",
        "cat <<\nssgrep search q\nEOF\n",
        "bash -c 'ssgrep search q",
    ],
)
def test_incomplete_shell_syntax_never_reports_a_partial_invocation(command):
    assert scan_shell("ssgrep search q").argvs == [("search", "q")]
    rejected = scan_shell(command)
    assert rejected.error == "unparseable_shell"
    assert rejected.argvs == []


@pytest.mark.parametrize("command", ["env -u", "env --chdir", "timeout -k", "xargs -n"])
def test_wrappers_missing_required_option_values_are_unsupported(command):
    positive = scan_shell("env -u SECRET ssgrep search q")
    assert positive.argvs == [("search", "q")]
    rejected = scan_shell(command)
    assert rejected.unsupported
    assert rejected.argvs == []


def test_scan_limits_are_distinct_from_parse_failure_and_keep_valid_request():
    from ssgrep.accounting.readers.shell_scan import MAX_COMMAND_CHARS, scan_words

    accepted = scan_words(["ssgrep", "search", "q"])
    assert accepted.argvs == [("search", "q")]
    assert accepted.error is None

    too_large = scan_words(["ssgrep", "search", "q" * MAX_COMMAND_CHARS])
    assert too_large.error == "command_too_large"
    assert too_large.argvs == []

    scan = scan_shell("# comment with ssgrep search hidden\nssgrep search visible")
    assert scan.argvs == [("search", "visible")]
    assert scan.operations == ("search",)


def test_logical_lines_drops_comments_and_heredoc_bodies_but_keeps_quoted_newlines():
    from ssgrep.accounting.readers.shell_scan import logical_lines

    text = "# note\nssgrep search 'a\nb'\ncat <<EOF\nssgrep hidden\nEOF\necho done"
    assert logical_lines(text) == ["", "ssgrep search 'a\nb'", "cat <<EOF", "echo done"]


def test_unparseable_nested_script_is_unsupported_not_a_partial_invocation():
    assert scan_shell("bash -c 'ssgrep search q'").argvs == [("search", "q")]
    rejected = scan_shell('bash -c "ssgrep search \'unterminated"')
    assert rejected.argvs == []
    assert rejected.unsupported
    assert rejected.has_other_commands
    assert rejected.error is None


def test_ssgrep_without_an_operation_is_other_command_not_an_invocation():
    positive = scan_shell("ssgrep search q")
    assert positive.argvs == [("search", "q")]
    assert not positive.has_other_commands
    flags_only = scan_shell("ssgrep --json")
    assert flags_only.argvs == []
    assert flags_only.has_other_commands


def test_env_assignments_are_consumed_before_the_request_and_bare_stdbuf_is_rejected():
    env = scan_shell("env -i A=1 B=2 ssgrep search q")
    assert env.argvs == [("search", "q")]
    assert not env.unsupported
    bare = scan_shell("stdbuf ssgrep search q")
    assert bare.argvs == []
    assert bare.unsupported

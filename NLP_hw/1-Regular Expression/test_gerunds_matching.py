"""Test gerunds-matching regular expression."""
import re

import pytest


# *** ADD YOUR PATTERN BELOW *** #
pattern = r"\b\w{2,6}ing"
#raise NotImplementedError("Add your pattern to the test file.")
# *** ADD YOUR PATTERN ABOVE *** #


test_cases = [
    ("harry loves to sing while showering",['showering']),
    ("singing is what he is doing", ['singing', 'doing']),
    ("oh, she is listening to an interesting story right now", ['listening']),
    ("peter really likes to cook", []),
    ("that usually happens in the evening", []),
    ("she prefers wearing one earing", ['wearing']),
    ("he is bargaining with her", ['bargaining']),
]


@pytest.mark.parametrize("string,matches", test_cases)
def test_name_matching(string, matches: list):
    """Test whether pattern correctly matches or does not match input."""
    assert re.findall(pattern, string) is not None
    assert re.findall(pattern, string) == matches
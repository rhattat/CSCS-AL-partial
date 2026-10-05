import pytest

from cscs_al_partial.strategies import get_cold_start_class, list_strategies
from cscs_al_partial.cold_start import CSCSColdStart, RandomColdStart, ProbCoverColdStart


def test_list_strategies_matches_registry_entries():
    strategies = list_strategies()
    assert set(strategies) == {"randomCS", "probcover", "cscs_curriculum"}


@pytest.mark.parametrize("name,expected_cls", [
    ("randomCS", RandomColdStart),
    ("probcover", ProbCoverColdStart),
    ("cscs_curriculum", CSCSColdStart),
])
def test_get_cold_start_class_returns_expected_class(name, expected_cls):
    assert get_cold_start_class(name) is expected_cls


def test_get_cold_start_class_unknown_name_raises():
    with pytest.raises(ValueError):
        get_cold_start_class("not_a_real_strategy")

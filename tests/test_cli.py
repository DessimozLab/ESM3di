import logging

import pytest

from esm3di.cli import setup_logging


@pytest.mark.parametrize(
    ("verbosity", "expected_level"),
    [
        (0, logging.CRITICAL + 1),
        (1, logging.ERROR),
        (2, logging.WARNING),
        (3, logging.INFO),
    ],
)
def test_setup_logging_maps_verbosity_levels(verbosity, expected_level):
    setup_logging(verbosity)

    logger = logging.getLogger("esm3di")
    assert logger.level == expected_level
    assert logger.propagate is False
    assert len(logger.handlers) == 1


def test_setup_logging_rejects_invalid_verbosity():
    with pytest.raises(ValueError, match="between 0 and 3"):
        setup_logging(4)

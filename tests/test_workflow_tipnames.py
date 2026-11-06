import pytest

from esm3di.workflow.scripts import tipnames


LONG_NAME = "FJJI_" + "x" * 120 + "_E_325-499_AlphaFold"


def test_placeholders_round_trip_long_names(tmp_path):
    names = ["a", LONG_NAME, "HPHV_virus_E1_1-10_AlphaFold"]
    name_map = tipnames.read_name_map(tipnames.write_name_map(names, tmp_path / "names.tsv"))
    short = tipnames.placeholder_names(names)
    assert all(len(name) <= 100 for name in short)

    newick = f"(({short[0]}:0.1,{short[1]}:0.2):0.05,{short[2]}:0.3);"
    assert tipnames.restore_names(newick, name_map) == (
        f"((a:0.1,{LONG_NAME}:0.2):0.05,HPHV_virus_E1_1-10_AlphaFold:0.3);"
    )


def test_restore_rejects_missing_tips(tmp_path):
    name_map = {"seq000000": "a", "seq000001": "b"}
    with pytest.raises(ValueError, match="placeholder"):
        tipnames.restore_names("(seq000000:0.1);", name_map)

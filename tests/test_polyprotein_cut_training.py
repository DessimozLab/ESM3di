from experiments.analyses.polyprotein_cut import run_comparison, train_all_data


def test_cut_site_parser_accepts_single_residue_and_range_annotations(tmp_path):
    source = tmp_path / "source.dat"
    source.write_text(
        'ID   TEST_VIRUS\n'
        'FT   SITE            3\n'
        'FT                   /note="Cleavage"\n'
        'FT   SITE            8..9\n'
        'FT                   /note="Cleavage"\n'
        'FT   CHAIN           1..12\n'
        '//\n'
    )

    assert run_comparison.parse_cut_sites(source) == {"TEST_VIRUS": [3, 8]}


def test_curated_training_data_adds_only_explicitly_labelled_entries():
    records, excluded = train_all_data.load_training_records()

    assert len(records) == 88
    assert sum(len(record["sites"]) for record in records) == 805
    added = {record["accession"]: record for record in records if record.get("source_kind") == "additional_entry"}
    assert set(added) == {"P19711", "Q01499"}
    assert added["P19711"]["sites"] == [497]
    assert added["Q01499"]["sites"] == [497]
    assert len(excluded) == 35
    assert all("no explicit SITE" in row["reason"] for row in excluded)

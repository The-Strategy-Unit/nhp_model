"""test nhp data (reference)"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name,missing-function-docstring
# ruff: noqa: PLR2004

from nhp.model.data import reference

expected_hsa_variants = {"lle", "hle", "ppp"}


def test_variants():
    # arrange

    # act
    vl = reference.variant_lookup()

    # assert
    assert len(vl) == 19
    assert set(vl.values()) == expected_hsa_variants


def test_life_expectancy():
    # arrange

    # act
    le = reference.life_expectancy()

    # assert
    assert len(le) == 276
    assert list(le.columns) == ["var", "sex", "age"] + [
        str(i) for i in range(2018, 2044)
    ]
    assert set(le["var"]) == expected_hsa_variants
    assert set(le["sex"]) == {1, 2}
    assert list(le["age"]) == list(range(55, 101)) * 6
    assert le[[str(i) for i in range(2018, 2043)]].sum().sum() == 89323.6


def test_split_normal_params():
    # arrange

    # act
    snp = reference.split_normal_params()

    # assert
    assert len(snp) == 144
    assert list(snp.columns) == [
        "var",
        "sex",
        "year",
        "mode",
        "sd1",
        "sd2",
    ]
    assert set(snp["var"]) == expected_hsa_variants
    assert set(snp["sex"]) == {"f", "m"}
    assert snp["year"].to_list() == list(range(2020, 2044)) * 6
    assert snp[["mode", "sd1", "sd2"]].sum().to_list() == [
        12.159496878354162,
        55.57842646603717,
        140.31508181965998,
    ]

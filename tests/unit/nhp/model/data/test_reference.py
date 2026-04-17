"""test nhp data (reference)."""

from unittest.mock import call

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
    le = reference.life_expectancy(2023, 2040)

    # assert
    assert len(le) == 144
    assert le.index.names == ["year", "sex", "age"]
    assert le.sum() == 2242.4


def test_hsa_metalog_parameters(mocker):
    # arrange
    m_metalog = mocker.patch("nhp.model.data.reference.Metalog", return_value="metalog")
    m_metalog_params = mocker.patch(
        "nhp.model.data.reference.MetalogParameters", return_value="metalog_params"
    )

    # act
    hsa_metalog_params = reference.hsa_metalog_parameters(2040)

    # assert
    assert hsa_metalog_params == {1: "metalog", 2: "metalog"}
    assert m_metalog.call_count == 2
    assert m_metalog_params.call_count == 2

    assert m_metalog.call_args_list[0] == call(
        "metalog_params",
        [
            0.22597766540977504,
            0.1216726601048689,
            0.8497745489369272,
            0.2322532816930789,
            -3.5019133822671478,
            0.20162460337715876,
            1.6358569059716728,
            -2.990760420935966,
            14.260971405049265,
        ],
    )

    assert m_metalog.call_args_list[1] == call(
        "metalog_params",
        [
            0.36636018884627675,
            0.251240039915706,
            1.1887234662388995,
            -0.2887848682168371,
            -5.14659756243718,
            -0.09486189481151404,
            2.586705415950949,
            -4.12765672185841,
            18.1202866074243,
        ],
    )

    assert m_metalog_params.call_args_list[0] == call(
        boundedness=4, method=1, lower_bound=0.0, upper_bound=100.0, num_terms=9
    )
    assert m_metalog_params.call_args_list[1] == call(
        boundedness=4, method=1, lower_bound=0.0, upper_bound=100.0, num_terms=9
    )

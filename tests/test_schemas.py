from app.models.schemas import AnalysisRequest, validate_ticker


def _rejects(v):
    try:
        validate_ticker(v)
    except ValueError:
        return True
    return False


def test_valid_tickers_normalized():
    assert validate_ticker(" 600519 ") == "600519"
    assert validate_ticker("sh600519") == "SH600519"
    assert validate_ticker("aapl") == "AAPL"
    assert validate_ticker("BRK.B") == "BRK.B"


def test_invalid_tickers_rejected():
    for bad in ["", "12345", "999999", "abc def", "600519;DROP", "1234567"]:
        assert _rejects(bad), bad


def test_weight_range_validated():
    try:
        AnalysisRequest(ticker="600519", fundamental_weight=1.5)
    except ValueError:
        return
    raise AssertionError("越界权重应被拒绝")

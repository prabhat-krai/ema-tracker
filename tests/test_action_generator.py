import pytest
from pathlib import Path
from src.action_generator import parse_log_file, compare_signals

def test_parse_log_file(tmp_path):
    log_content = """2026-02-21 17:01:46 | INFO | ✅ BULLISH      | AAPL            | $    264.58 | Above all weekly EMAs
2026-02-21 17:01:48 | INFO | 🔴 EXIT       | MSFT            | $    397.23 | Below 40W EMA
2026-02-21 17:01:50 | INFO | 🟢 HOLD_ADD   | NVDA            | $    189.82 | Above all weekly EMAs
2026-02-21 17:01:52 | INFO | 🟡 WAIT       | META            | $    655.66 | EMAs converging, no breakout yet
2026-02-21 17:02:03 | INFO | 🟠 CAUTIOUS   | AVGO            | $    332.65 | Below 20W EMA
2026-02-21 17:02:16 | INFO | 🟣 FADING     | LLY             | $   1009.52 | Below 10W EMA - momentum fading
"""
    log_file = tmp_path / "test.log"
    log_file.write_text(log_content)

    signals = parse_log_file(log_file)
    assert len(signals) == 6
    assert signals["AAPL"] == "BULLISH"
    assert signals["MSFT"] == "EXIT"
    assert signals["NVDA"] == "HOLD_ADD"
    assert signals["META"] == "WAIT"
    assert signals["AVGO"] == "CAUTIOUS"
    assert signals["LLY"] == "FADING"

def test_compare_signals():
    old = {
        "AAPL": "WAIT",
        "MSFT": "HOLD_ADD",
        "NVDA": "BULLISH",
        "META": "EXIT",
        "AVGO": "CAUTIOUS",
        "LLY": "HOLD_ADD",
        "TSLA": "WAIT", # No change
    }
    
    new = {
        "AAPL": "BULLISH",  # NEW BUY
        "MSFT": "EXIT",     # NEW SELL
        "NVDA": "CAUTIOUS", # DOWNGRADE
        "META": "WAIT",     # UPGRADE
        "AVGO": "HOLD_ADD", # UPGRADE
        "LLY": "FADING",    # DOWNGRADE
        "TSLA": "WAIT",     # No change
        "NEW_STK": "BULLISH", # Not in old log
    }

    transitions = compare_signals(old, new)
    
    assert len(transitions) == 6
    # Let's verify specific actions
    actions = {t["Symbol"]: t["Action Category"] for t in transitions}
    
    assert "🚀 NEW BUY (Action: Buy Now)" in actions["AAPL"]
    assert "🚨 NEW SELL (Action: Sell Now)" in actions["MSFT"]
    assert "⚠️ DOWNGRADE" in actions["NVDA"]
    assert "⚠️ DOWNGRADE" in actions["LLY"]
    assert "📈 UPGRADE" in actions["META"]
    assert "📈 UPGRADE" in actions["AVGO"]
    assert "TSLA" not in actions # Should skip unchanged
    assert "NEW_STK" not in actions # Should skip new stocks without history

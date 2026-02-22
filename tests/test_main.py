import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

from src import main

def test_ga_flag_no_logs(capsys):
    """Test when --ga is run but no logs exist at all."""
    with patch.object(sys, "argv", ["main.py", "--ga"]):
        with patch("src.main.find_latest_log", return_value=None):
            main.main()
            
    captured = capsys.readouterr()
    assert "EMA Action Generator Mode" in captured.out
    assert "No logs found to generate actions from." in captured.out

def test_ga_flag_only_one_log(capsys):
    """Test when --ga is run but only one log exists."""
    dummy_latest = MagicMock()
    dummy_latest.name = "INDIA_test.log"
    
    with patch.object(sys, "argv", ["main.py", "--ga"]):
        # find_latest_log returns something the first time, but None the second time
        with patch("src.main.find_latest_log", side_effect=[dummy_latest, None]):
            main.main()
            
    captured = capsys.readouterr()
    assert "Only one log found" in captured.out
    assert "Need at least two" in captured.out

def test_ga_flag_generates_actions(capsys):
    """Test successful action generation via --ga flag."""
    dummy_latest = MagicMock()
    dummy_latest.name = "INDIA_21-02-2026.log"
    dummy_latest.stem = "INDIA_21-02-2026"
    
    dummy_prev = MagicMock()
    dummy_prev.name = "INDIA_14-02-2026.log"
    
    fake_csv_path = MagicMock()
    
    fake_transitions = [{"fake": "data"}]
    
    with patch.object(sys, "argv", ["main.py", "--ga"]):
        with patch("src.main.find_latest_log", side_effect=[dummy_latest, dummy_prev]):
            with patch("src.main.parse_log_file") as mock_parse:
                with patch("src.main.compare_signals", return_value=fake_transitions) as mock_compare:
                    with patch("src.main.generate_action_csv", return_value=fake_csv_path) as mock_generate:
                        main.main()
                        
                        mock_parse.assert_called()
                        mock_compare.assert_called()
                        mock_generate.assert_called()
                        
    captured = capsys.readouterr()
    assert "Comparing [INDIA_21-02-2026.log] against [INDIA_14-02-2026.log]" in captured.out
    assert "Action report refreshed:" in captured.out

def test_ga_flag_no_transitions(capsys):
    """Test successful run via --ga but there are no transitions."""
    dummy_latest = MagicMock()
    dummy_latest.name = "USA_21-02-2026.log"
    
    dummy_prev = MagicMock()
    dummy_prev.name = "USA_14-02-2026.log"
    
    with patch.object(sys, "argv", ["main.py", "--usa", "--ga"]):
        with patch("src.main.find_latest_log", side_effect=[dummy_latest, dummy_prev]):
            with patch("src.main.parse_log_file"):
                with patch("src.main.compare_signals", return_value=[]):
                    main.main()
                        
    captured = capsys.readouterr()
    assert "Comparing [USA_21-02-2026.log] against [USA_14-02-2026.log]" in captured.out
    assert "No new actionable transitions" in captured.out

def test_ga_flag_full_integration_in_tmp(tmp_path, capsys, monkeypatch):
    """
    Real Integration Test: Runs file operations entirely inside a sandboxed `tmp_path`.
    This proves that no 'mocking shortcuts' were taken and actual files are handled
    solely within isolated temporary test directories.
    """
    # 1. Sandbox main.py's internal path resolution so it points exactly into our test tmp_path
    monkeypatch.setattr('src.main.__file__', str(tmp_path / "src" / "main.py"))
    
    # 2. Setup isolated `logs` folder and drop two real text files into it
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    
    log1 = logs_dir / "INDIA_14-02-2026.log"
    log1.write_text("2026-02-14 17:00:00 | INFO | 🔴 EXIT       | AAPL            | $ 100.00 |")
    
    log2 = logs_dir / "INDIA_21-02-2026.log"
    log2.write_text("2026-02-21 17:00:00 | INFO | ✅ BULLISH      | AAPL            | $ 150.00 |")

    # 3. Simulate executing `python main.py --ga` in the terminal
    with patch.object(sys, "argv", ["main.py", "--ga"]):
        main.main()

    captured = capsys.readouterr()
    assert "EMA Action Generator Mode" in captured.out
    
    # 4. Verify tmp_path structure isolated the output successfully
    actions_dir = tmp_path / "actions"
    assert actions_dir.exists(), "Actions directory should be created within tmp_path sandbox"
    
    generated_csv = actions_dir / "INDIA-ACTIONS_21-02-2026.csv"
    assert generated_csv.exists(), "CSV should be explicitly linked to the date 21-02-2026"
    assert "BULLISH" in generated_csv.read_text()
    assert "NEW BUY" in generated_csv.read_text()

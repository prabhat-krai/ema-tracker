"""
Unit Tests for Quantitative Momentum Screener CLI Runner.
"""

from pathlib import Path
import pytest

from src.momentum.cli import build_arg_parser, format_console_summary, main
from src.momentum.engine import MomentumPipeline


class TestMomentumCLI:
    """Test suite for CLI argument parser and execution."""

    def test_build_arg_parser_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args([])

        assert args.universe == "india"
        assert args.top == 10
        assert args.weighting == "inv_vol"
        assert args.capital == 1_000_000.0
        assert args.holdings is None
        assert args.rank_buffer == 20
        assert args.min_weight == 0.05
        assert args.max_weight == 0.20
        assert args.deadband == 0.02
        assert args.export == "all"
        assert args.offline is False

    def test_build_arg_parser_custom_args(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--universe", "usa",
            "--top", "15",
            "--weighting", "bounded_parity",
            "--capital", "500000",
            "--holdings", "AAPL,MSFT,NVDA",
            "--rank-buffer", "25",
            "--min-weight", "0.04",
            "--max-weight", "0.25",
            "--deadband", "0.03",
            "--export", "json",
            "--offline",
            "--verbose",
        ])

        assert args.universe == "usa"
        assert args.top == 15
        assert args.weighting == "bounded_parity"
        assert args.capital == 500_000.0
        assert args.holdings == "AAPL,MSFT,NVDA"
        assert args.rank_buffer == 25
        assert args.min_weight == 0.04
        assert args.max_weight == 0.25
        assert args.deadband == 0.03
        assert args.export == "json"
        assert args.offline is True
        assert args.verbose is True

    def test_cli_main_offline_india_success(self, tmp_path):
        out_dir = tmp_path / "cli_reports"
        exit_code = main([
            "--offline",
            "--universe", "india",
            "--top", "5",
            "--weighting", "equal",
            "--output-dir", str(out_dir),
            "--export", "all",
            "--quiet",
        ])

        assert exit_code == 0
        assert (out_dir / "portfolio.csv").exists()
        assert (out_dir / "portfolio.json").exists()
        assert (out_dir / "portfolio_summary.md").exists()

    def test_cli_main_offline_usa_bounded_parity_with_holdings(self, tmp_path):
        out_dir = tmp_path / "usa_reports"
        exit_code = main([
            "--offline",
            "--universe", "usa",
            "--top", "8",
            "--weighting", "bounded_parity",
            "--holdings", "AAPL,MSFT",
            "--output-dir", str(out_dir),
            "--export", "csv",
            "--quiet",
        ])

        assert exit_code == 0
        assert (out_dir / "usa_momentum.csv").exists()

    def test_cli_main_invalid_args_returns_error(self):
        with pytest.raises(SystemExit):
            main(["--universe", "mars"])

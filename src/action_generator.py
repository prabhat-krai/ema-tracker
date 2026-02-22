import csv
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# State categorizations for transitions
BUY_SIGNALS = {"BULLISH"}
SELL_SIGNALS = {"EXIT"}
HOLD_SIGNALS = {"HOLD_ADD"}
WATCH_SIGNALS = {"WAIT"}
CAUTION_SIGNALS = {"CAUTIOUS", "FADING"}

def parse_log_file(filepath: Path) -> dict[str, str]:
    """
    Reads a log file and extracts the Symbol -> Signal mapping.
    Assumes log lines containing signals look like:
    2026-02-21 17:03:04 | INFO | ✅ BULLISH      | AAPL            | $    238.25 | ...
    """
    signals = {}
    if not filepath.exists():
        return signals

    # Regex to match the signal format line
    # Groups: 1: Signal (e.g. BULLISH), 2: Symbol (e.g. AAPL)
    signal_pattern = re.compile(
        r"\|\s*(?:✅|🔴|🟠|🟣|🟢|🟡)\s+([A-Z_]+)\s*\|\s*([A-Z0-9.\-]+)\s*\|"
    )

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                match = signal_pattern.search(line)
                if match:
                    signal = match.group(1).strip()
                    symbol = match.group(2).strip()
                    signals[symbol] = signal
    except Exception as e:
        logger.error(f"Failed to parse log file {filepath}: {e}")

    return signals


def find_latest_log(log_dir: Path, market_prefix: str, exclude_file: Path) -> Path | None:
    """
    Finds the most recent log file for the given market, excluding the current one.
    """
    pattern = f"{market_prefix}_*.log"
    log_files = list(log_dir.glob(pattern))
    
    # Sort files chronologically by parsing the DD-MM-YYYY date from the filename
    def extract_date(filepath: Path):
        try:
            date_str = filepath.stem.split("_")[-1]
            return datetime.strptime(date_str, "%d-%m-%Y")
        except ValueError:
            return datetime.min

    log_files = sorted(log_files, key=extract_date, reverse=True)
    
    for log_file in log_files:
        if log_file.resolve() != exclude_file.resolve():
            return log_file
            
    return None


def compare_signals(old_signals: dict[str, str], new_signals: dict[str, str]) -> list[dict]:
    """
    Compares old and new signals and categorizes transitions.
    Returns a list of dictionaries with transition details.
    """
    transitions = []

    for symbol, new_signal in new_signals.items():
        old_signal = old_signals.get(symbol)
        
        # If the stock wasn't in the previous log or hasn't changed, skip
        if not old_signal or old_signal == new_signal:
            continue

        action_category = None
        notes = f"Changed from {old_signal} to {new_signal}"

        # 1. NEW BUY
        if new_signal in BUY_SIGNALS and old_signal not in BUY_SIGNALS:
            action_category = "🚀 NEW BUY (Action: Buy Now)"
        
        # 2. NEW SELL
        elif new_signal in SELL_SIGNALS and old_signal not in SELL_SIGNALS:
            action_category = "🚨 NEW SELL (Action: Sell Now)"
            
        # 3. DOWNGRADE
        elif new_signal in CAUTION_SIGNALS and old_signal in (HOLD_SIGNALS | BUY_SIGNALS):
            action_category = "⚠️ DOWNGRADE (Action: Caution/Trim)"
            
        # 4. UPGRADE
        elif new_signal in WATCH_SIGNALS and old_signal in SELL_SIGNALS:
            action_category = "📈 UPGRADE (Action: Watch closely)"
        elif new_signal in HOLD_SIGNALS and old_signal in CAUTION_SIGNALS:
            action_category = "📈 UPGRADE (Action: Accumulate/Hold)"

        if action_category:
            transitions.append({
                "Symbol": symbol,
                "Previous Signal": old_signal,
                "Current Signal": new_signal,
                "Action Category": action_category,
                "Notes": notes
            })

    # Sort transitions by Action Category to group them
    transitions.sort(key=lambda x: x["Action Category"])
    return transitions


def generate_action_csv(transitions: list[dict], output_dir: Path, market_prefix: str, target_date_str: str = None) -> Path | None:
    """
    Writes the transitions to a CSV file.
    If target_date_str is not provided, defaults to today's date.
    """
    if not transitions:
        logger.info(f"No actionable transitions found for {market_prefix}.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the target date provided, otherwise fallback to today
    date_to_use = target_date_str if target_date_str else datetime.now().strftime('%d-%m-%Y')
    csv_path = output_dir / f"{market_prefix}-ACTIONS_{date_to_use}.csv"
    
    fieldnames = ["Symbol", "Previous Signal", "Current Signal", "Action Category", "Notes"]

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(transitions)
        logger.info(f"Action report generated at: {csv_path}")
        return csv_path
    except Exception as e:
        logger.error(f"Failed to write CSV report {csv_path}: {e}")
        return None

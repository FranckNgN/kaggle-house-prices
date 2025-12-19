import json
from pathlib import Path
from typing import Any, Dict, Optional
import config_local.local_config as config

def update_engineering_summary(stage_name: str, details: Dict[str, Any], summary_path: Optional[str] = None) -> None:
    """
    Update the global engineering summary with details from a specific stage.
    
    Args:
        stage_name: Name of the preprocessing stage
        details: Dictionary of details to add for this stage
        summary_path: Path to the summary JSON file (defaults to config.RUNS_DIR / "current_engineering_summary.json")
    """
    if summary_path is None:
        summary_path = str(config.RUNS_DIR / "current_engineering_summary.json")
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    if path.exists():
        try:
            with open(path, "r") as f:
                summary = json.load(f)
        except Exception:
            pass
    
    summary[stage_name] = details
    
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"    [Manifest] Updated {stage_name} engineering details.")

def reset_engineering_summary(summary_path: Optional[str] = None) -> None:
    """
    Clear the engineering summary (usually at the start of a pipeline run).
    
    Args:
        summary_path: Path to the summary JSON file (defaults to config.RUNS_DIR / "current_engineering_summary.json")
    """
    if summary_path is None:
        summary_path = str(config.RUNS_DIR / "current_engineering_summary.json")
    path = Path(summary_path)
    if path.exists():
        path.unlink()
    print("    [Manifest] Reset engineering summary.")


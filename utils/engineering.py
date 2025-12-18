import json
from pathlib import Path
from typing import Any, Dict

def update_engineering_summary(stage_name: str, details: Dict[str, Any], summary_path: str = "runs/current_engineering_summary.json") -> None:
    """
    Update the global engineering summary with details from a specific stage.
    """
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

def reset_engineering_summary(summary_path: str = "runs/current_engineering_summary.json") -> None:
    """Clear the engineering summary (usually at the start of a pipeline run)."""
    path = Path(summary_path)
    if path.exists():
        path.unlink()
    print("    [Manifest] Reset engineering summary.")


#!/usr/bin/env python
"""
Helper script to prepare code for Kaggle execution.
Checks git status and provides guidance for syncing code to Kaggle.
"""
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_git_status():
    """Check git status and return information."""
    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        
        if result.returncode != 0:
            return None, "Not a git repository or git not found"
        
        # Check for uncommitted changes
        uncommitted = result.stdout.strip()
        has_changes = bool(uncommitted)
        
        # Get current branch
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
        
        # Check if branch is pushed
        unpushed_result = subprocess.run(
            ["git", "log", f"origin/{current_branch}..HEAD", "--oneline"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        has_unpushed = bool(unpushed_result.stdout.strip()) if unpushed_result.returncode == 0 else False
        
        return {
            "has_changes": has_changes,
            "uncommitted": uncommitted,
            "current_branch": current_branch,
            "has_unpushed": has_unpushed
        }, None
        
    except FileNotFoundError:
        return None, "Git not found. Please install Git."
    except Exception as e:
        return None, f"Error checking git status: {e}"


def get_repo_url():
    """Get the remote repository URL."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def main():
    """Main function."""
    print("=" * 70)
    print("KAGGLE SYNC HELPER")
    print("=" * 70)
    print()
    
    # Check git status
    print("Checking git status...")
    status, error = check_git_status()
    
    if error:
        print(f"[WARNING] {error}")
        print()
        print("To use Kaggle GPU workflow:")
        print("1. Initialize git repository: git init")
        print("2. Add remote: git remote add origin <your-repo-url>")
        print("3. Commit your code: git add . && git commit -m 'Initial commit'")
        print("4. Push to remote: git push -u origin main")
        return
    
    repo_url = get_repo_url()
    
    print(f"Current branch: {status['current_branch']}")
    if repo_url:
        print(f"Repository: {repo_url}")
    print()
    
    # Check for uncommitted changes
    if status['has_changes']:
        print("[WARNING] You have uncommitted changes:")
        print("-" * 70)
        print(status['uncommitted'])
        print("-" * 70)
        print()
        print("Before syncing to Kaggle:")
        print("1. Review your changes")
        print("2. Commit: git add . && git commit -m 'Your commit message'")
        print("3. Push: git push")
    else:
        print("[OK] No uncommitted changes")
    
    print()
    
    # Check for unpushed commits
    if status['has_unpushed']:
        print("[WARNING] You have commits that haven't been pushed:")
        print("  Run: git push")
    else:
        print("[OK] All commits are pushed to remote")
    
    print()
    print("=" * 70)
    print("NEXT STEPS FOR KAGGLE")
    print("=" * 70)
    print()
    
    if status['has_changes'] or status['has_unpushed']:
        print("1. Commit and push your changes first (see above)")
        print()
    
    print("2. Open/create Kaggle Notebook")
    print("   - Go to: https://www.kaggle.com/code")
    print("   - Create new notebook or open existing one")
    print("   - Enable GPU: Settings -> Accelerator -> GPU (P100 or T4)")
    print()
    print("3. Add competition dataset")
    print("   - Click 'Add Data' in notebook")
    print("   - Search for 'house-prices-advanced-regression-techniques'")
    print("   - Click 'Add'")
    print()
    
    if repo_url:
        print("4. In the Kaggle notebook, clone your repository:")
        print(f"   !git clone {repo_url} /kaggle/working/project")
        print()
        print("   Or use the template notebook:")
        print(f"   kaggle/notebooks/kaggle_gpu_runner.ipynb")
        print("   (Update REPO_URL in Cell 1)")
    else:
        print("4. Set up your repository URL in the Kaggle notebook")
        print("   (Edit Cell 1 in kaggle_gpu_runner.ipynb)")
    
    print()
    print("5. Run the notebook cells to:")
    print("   - Clone repository")
    print("   - Install dependencies")
    print("   - Setup environment")
    print("   - Train models on GPU")
    print()
    print("=" * 70)
    print("[INFO] For detailed workflow guide, see: docs/HYBRID_WORKFLOW.md")
    print("=" * 70)


if __name__ == "__main__":
    main()


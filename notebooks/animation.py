"""
NFL Play Animation Script (Refactored)

This script now uses the PlayAnimator class from the nfl_analysis package.
It provides a simple interface for animating plays using consolidated data.

Usage:
    python animation.py <game_id> <play_id>

Example:
    python animation.py 2023090700 1679
"""

import sys
from pathlib import Path

# Add parent directory to path to import from package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_analysis import PlayAnimator


def main():
    """Main entry point for animation script."""
    # Check command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python animation.py <game_id> <play_id>")
        print("Example: python animation.py 2023090700 1679")
        sys.exit(1)

    print(f"Script name: {sys.argv[0]}")
    print("Arguments provided:")
    print(f"  Game ID: {sys.argv[1]}")
    print(f"  Play ID: {sys.argv[2]}")

    # Parse arguments
    try:
        game_id = int(sys.argv[1].strip())
        play_id = int(sys.argv[2].strip())
    except ValueError as e:
        print(f"Error: Game ID and Play ID must be integers. {e}")
        sys.exit(1)

    # Determine data directory
    # Check both old path (for backward compatibility) and new path
    script_dir = Path(__file__).parent
    old_data_dir = script_dir.parent / "consolidated_data"
    new_data_dir = script_dir.parent / "data" / "consolidated"

    if new_data_dir.exists():
        data_dir = str(new_data_dir)
    elif old_data_dir.exists():
        data_dir = str(old_data_dir)
    else:
        print("Error: Could not find consolidated data directory.")
        print(f"Tried: {new_data_dir}")
        print(f"Tried: {old_data_dir}")
        sys.exit(1)

    print(f"Using data directory: {data_dir}")

    try:
        # Create animator using the package
        print("\nCreating animator...")
        animator = PlayAnimator(data_dir=data_dir)

        # Create animation
        print(f"Loading play data for Game {game_id}, Play {play_id}...")
        ani = animator.create_animation(
            game_id=game_id,
            play_id=play_id,
            interval=200,  # 200ms between frames
            repeat=False
        )

        # Display animation
        print("Displaying animation (will show for 30 seconds)...")
        animator.show(ani, pause_time=30)

        print("\nAnimation complete!")
        return 0

    except FileNotFoundError as e:
        print(f"\nError: Could not find data for the specified play.")
        print(f"Details: {e}")
        return 1
    except KeyError as e:
        print(f"\nError: Missing required data column.")
        print(f"Details: {e}")
        return 1
    except Exception as e:
        print(f"\nError creating animation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

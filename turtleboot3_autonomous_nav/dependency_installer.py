"""Explicit installer for the TurtleBot3 source dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


# The upstream ``turtlebot3`` repository also contains an optional
# turtlebot3_cartographer package. Cartographer is not released for Jazzy and
# is unrelated to the Stage 4 Gazebo launcher, so never ask rosdep to resolve
# every package in that repository.
STAGE4_SOURCE_PATHS = (
    'src/turtlebot3_msgs',
    'src/turtlebot3/turtlebot3_description',
    'src/turtlebot3_simulations/turtlebot3_gazebo',
)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def install(workspace: Path, manifest: Path) -> None:
    """Import, resolve and build the three pinned TurtleBot3 repositories."""
    source_directory = workspace / 'src'
    source_directory.mkdir(parents=True, exist_ok=True)
    with manifest.open('rb') as stream:
        print('+ vcs import --force ' + str(source_directory), flush=True)
        subprocess.run(['vcs', 'import', '--force', str(source_directory)], stdin=stream, check=True)
    run([
        'rosdep', 'install', '--from-paths', *STAGE4_SOURCE_PATHS,
        '--ignore-src', '-r', '-y',
    ], cwd=workspace)
    run([
        'colcon', 'build', '--symlink-install', '--packages-up-to',
        'turtlebot3_gazebo',
    ], cwd=workspace)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.home() / 'tb3_dependencies')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args(argv)
    manifest = args.manifest or Path(__file__).resolve().parents[1] / 'dependencies' / 'turtlebot3_jazzy.repos'
    try:
        install(args.workspace.expanduser(), manifest.expanduser().resolve())
    except subprocess.CalledProcessError as error:
        print(f'Dependency installation failed with exit status {error.returncode}.', file=sys.stderr)
        return error.returncode or 1
    print(f'Installed TurtleBot3 dependencies in {args.workspace.expanduser()}.')
    print(f'Source {args.workspace.expanduser() / "install" / "setup.bash"} before launching the mission.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Run psm.py with multiple seeds and record stdout/stderr in one log."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=list(range(6, 100)))
    parser.add_argument('--script', default='psm.py')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    script = Path(args.script).resolve()
    if not script.is_file():
        parser.error(f'Script not found: {script}. Run from the folder containing psm.py.')
    output = Path(args.output or f'psm_seeds_{datetime.now():%Y%m%d_%H%M%S_%f}.txt').resolve()
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    failures = []
    with output.open('w', encoding='utf-8', buffering=1) as log:
        def emit(text):
            log.write(text)
            log.flush()
            print(text, end='', flush=True)
        emit(f'Script: {script}\nPython: {sys.executable}\nSeeds: {args.seeds}\n')
        for seed in args.seeds:
            emit(f'\n{"=" * 24} SEED = {seed} {"=" * 24}\n')
            command = [sys.executable, '-u', str(script), '--seed', str(seed)]
            with subprocess.Popen(command, cwd=script.parent, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding='utf-8', errors='replace',
                                  bufsize=1) as process:
                try:
                    for line in process.stdout:
                        emit(line)
                    code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    emit(f'\nInterrupted during seed {seed}. Partial output saved.\n')
                    raise
            emit(f'\n[seed={seed}, exit_code={code}]\n')
            if code != 0:
                failures.append(seed)
        emit(f'\nFinished. Failed seeds: {failures}\n')
    print(f'Log saved to: {output}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())

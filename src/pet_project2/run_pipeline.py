import subprocess
import sys
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = WORK_DIR / 'src' / 'pet_project2'

SCRIPTS = [
    SCRIPTS_DIR / 'validation.py',
    SCRIPTS_DIR / 'train.py',
]


def main():
    for script in SCRIPTS:
        if not script.is_file():
            raise FileNotFoundError(f'Скрипт не найден: {script}')

        print(f'\nЗапускаю: {script.name}', flush=True)

        subprocess.run(
            [sys.executable, str(script)],
            cwd=WORK_DIR,
            check=True,
        )

        print(f'Завершён: {script.name}', flush=True)

    print('\nВесь пайплайн успешно выполнен.')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(
            f'\nПайплайн остановлен: {exc.cmd[-1]} завершился '
            f'с кодом ошибки {exc.returncode}.',
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode)
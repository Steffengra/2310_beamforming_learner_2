
from datetime import datetime


def progress_printer(
        progress,
        real_time_start,
) -> None:
    timedelta = datetime.now() - real_time_start
    finish_time = real_time_start + timedelta / progress

    print(f'\rSimulation completed: {progress:.2%}, '
          f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}', end='')

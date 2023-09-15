
import logging
from datetime import datetime


def progress_printer(
        progress: float,
        real_time_start: datetime,
        logger: logging.Logger = None,
) -> None:
    """Print simulation progress and estimated finish time using print() or a logger."""

    timedelta = datetime.now() - real_time_start
    finish_time = real_time_start + timedelta / progress

    result_string = (
        f'Simulation completed: {progress:.2%}, '
        f'est. finish {finish_time.year}-{finish_time.month:02d}-{finish_time.day:02d} '
        f'{finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}'
    )

    if logger is None:
        print('\r' + result_string, end='')
    else:
        logger.debug(result_string)

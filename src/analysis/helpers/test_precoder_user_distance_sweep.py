
from datetime import datetime
from pathlib import Path
import gzip
import pickle

import numpy as np
from matplotlib.pyplot import show as plt_show

import src
from src.data.satellite_manager import (
    SatelliteManager,
)
from src.data.user_manager import (
    UserManager,
)
from src.utils.plot_sweep import (
    plot_sweep,
)
from src.utils.profiling import (
    start_profiling,
    end_profiling,
)
from src.utils.progress_printer import (
    progress_printer,
)
from src.utils.update_sim import (
    update_sim,
)


def test_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range,
    precoder_name: str,
    get_precoder_func,
    calc_sum_rate_func,
) -> None:
    """
    Calculate the sum rates that a given precoder achieves for a given config
    over a given range of inter-user-distances with no channel error
    """

    def progress_print() -> None:
        progress = (distance_sweep_idx + 1) / (len(distance_sweep_range))
        progress_printer(progress=progress, real_time_start=real_time_start)

    def save_results():
        name = f'testing_{precoder_name}_sweep_{round(distance_sweep_range[0])}_{round(distance_sweep_range[-1])}.gzip'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'distance_sweep')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump([distance_sweep_range, metrics], file=file)

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)

    real_time_start = datetime.now()

    profiler = None
    if config.profile:
        profiler = start_profiling()

    metrics = {
        'sum_rate': {
            'mean': np.zeros(len(distance_sweep_range)),
            'std': np.zeros(len(distance_sweep_range)),
        },
    }

    for distance_sweep_idx, distance_sweep_value in enumerate(distance_sweep_range):

        config.user_dist_average = distance_sweep_value
        config.user_dist_bound = 0

        config.config_error_model.set_zero_error()

        update_sim(config, satellite_manager, user_manager)

        w_precoder = get_precoder_func(
            config=config,
            satellite_manager=satellite_manager
        )
        sum_rate = calc_sum_rate_func(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_precoder,
            noise_power_watt=config.noise_power_watt,
        )

        metrics['sum_rate']['mean'][distance_sweep_idx] = sum_rate
        metrics['sum_rate']['std'][distance_sweep_idx] = 0

        if distance_sweep_idx % 10 == 0:
            progress_print()

    if profiler is not None:
        end_profiling(profiler)

    save_results()

    plot_sweep(
        x=distance_sweep_range,
        y=metrics['sum_rate']['mean'],
        xlabel='User_dist',
        ylabel='Mean Sum Rate',
        yerr=metrics['sum_rate']['std'],
    )

    if config.show_plots:
        plt_show()


import numpy as np
from datetime import (
    datetime,
)
from pathlib import (
    Path,
)
from gzip import (
    open as gzip_open,
)
from pickle import (
    dump as pickle_dump,
)
from matplotlib.pyplot import (
    show as plt_show,
)

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


def test_precoder_error_sweep(
    config: 'src.config.config.Config',
    csit_error_sweep_range: np.ndarray,
    precoder_name: str,
    monte_carlo_iterations: int,
    get_precoder_func,
    calc_sum_rate_func,
) -> None:

    def progress_print() -> None:
        progress = (error_sweep_idx * monte_carlo_iterations + iter_idx + 1) / (len(csit_error_sweep_range) * monte_carlo_iterations)
        progress_printer(progress=progress, real_time_start=real_time_start)

    def set_new_error_value() -> None:
        if config.error_model.error_model_name == 'err_mult_on_steering_cos':
            config.error_model.uniform_error_interval['low'] = -1 * error_sweep_value
            config.error_model.uniform_error_interval['high'] = error_sweep_value
        elif config.error_model.error_model_name == 'err_sat2userdist':
            config.error_model.distance_error_std = error_sweep_value
        elif config.error_model.error_model_name == 'err_satpos_and_userpos':
            # todo: this model has 2 params
            # config.error_model.uniform_error_interval['low'] = -1 * error_sweep_value
            # config.error_model.uniform_error_interval['high'] = error_sweep_value
            config.error_model.phase_sat_error_std = error_sweep_value

        else:
            raise ValueError('Unknown error model name')

    def save_results():
        name = f'testing_{precoder_name}_sweep_{csit_error_sweep_range[0]}_{csit_error_sweep_range[-1]}_userwiggle_{config.user_dist_bound}.gzip'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, config.error_model.error_model_name, 'error_sweep')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip_open(Path(results_path, name), 'wb') as file:
            pickle_dump([csit_error_sweep_range, metrics], file=file)

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)

    real_time_start = datetime.now()

    profiler = None
    if config.profile:
        profiler = start_profiling()

    metrics = {
        'sum_rate': {
            'mean': np.zeros(len(csit_error_sweep_range)),
            'std': np.zeros(len(csit_error_sweep_range)),
        },
    }

    for error_sweep_idx, error_sweep_value in enumerate(csit_error_sweep_range):

        # set new error value
        set_new_error_value()

        # set up per monte carlo metrics
        sum_rate_per_monte_carlo = np.zeros(monte_carlo_iterations)

        for iter_idx in range(monte_carlo_iterations):

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

            # log results
            sum_rate_per_monte_carlo[iter_idx] = sum_rate

            if iter_idx % 50 == 0:
                progress_print()

        metrics['sum_rate']['mean'][error_sweep_idx] = np.mean(sum_rate_per_monte_carlo)
        metrics['sum_rate']['std'][error_sweep_idx] = np.std(sum_rate_per_monte_carlo)

    if profiler is not None:
        end_profiling(profiler)

    save_results()

    plot_sweep(
        x=csit_error_sweep_range,
        y=metrics['sum_rate']['mean'],
        yerr=metrics['sum_rate']['std'],
        xlabel='error value',
        ylabel='sum rate',
        title=precoder_name,
    )

    if config.show_plots:
        plt_show()


from numpy import (
    arange,
    zeros,
    mean,
    std,
)
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

from src.config.config import (
    Config,
)
from src.data.satellites import (
    Satellites,
)
from src.data.users import (
    Users,
)
from src.data.channel.los_channel_model import (
    los_channel_model,
)
from src.data.precoder.mmse_precoder import (
    mmse_precoder_normalized,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)
from src.utils.plot_sweep import (
    plot_sweep,
)

# TODO: move this to proper place
import matplotlib.pyplot as plt


def test_mmse_precoder_error_sweep(
        config,
        monte_carlo_iterations,
        csit_error_sweep_range,
) -> None:

    def progress_print() -> None:
        progress = (error_sweep_idx * monte_carlo_iterations + iter_idx + 1) / (len(csit_error_sweep_range) * monte_carlo_iterations)
        timedelta = datetime.now() - real_time_start
        finish_time = real_time_start + timedelta / progress

        print(f'\rSimulation completed: {progress:.2%}, '
              f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}', end='')

    def sim_update():
        users.update_positions(config=config)
        satellites.update_positions(config=config)

        satellites.calculate_satellite_distances_to_users(users=users.users)
        satellites.calculate_satellite_aods_to_users(users=users.users)
        satellites.calculate_steering_vectors_to_users(users=users.users)
        satellites.update_channel_state_information(channel_model=los_channel_model, users=users.users)
        satellites.update_erroneous_channel_state_information(error_model_config=config.error_model, users=users.users)

    def save_results():
        name = f'testing_mmse_sweep_{csit_error_sweep_range[0]}_{csit_error_sweep_range[-1]}_userwiggle_{config.user_dist_variance}.gzip'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'error_sweep')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip_open(Path(results_path, name), 'wb') as file:
            pickle_dump([csit_error_sweep_range, metrics], file=file)

    satellites = Satellites(config=config)
    users = Users(config=config)

    real_time_start = datetime.now()
    if config.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    metrics = {
        'sum_rate': {
            'mmse': {
                'mean': zeros(len(csit_error_sweep_range)),
                'std': zeros(len(csit_error_sweep_range)),
            },
        },
    }

    for error_sweep_idx, error_sweep_value in enumerate(csit_error_sweep_range):
        config.error_model.uniform_error_interval['low'] = -1 * error_sweep_value
        config.error_model.uniform_error_interval['high'] = error_sweep_value

        sum_rate_per_monte_carlo = zeros(monte_carlo_iterations)
        for iter_idx in range(monte_carlo_iterations):

            sim_update()

            w_mmse = mmse_precoder_normalized(
                channel_matrix=satellites.erroneous_channel_state_information,
                **config.mmse_args,
            )
            sum_rate = calc_sum_rate(
                channel_state=satellites.channel_state_information,
                w_precoder=w_mmse,
                noise_power_watt=config.noise_power_watt
            )
            sum_rate_per_monte_carlo[iter_idx] = sum_rate

            if iter_idx % 50 == 0:
                progress_print()

        metrics['sum_rate']['mmse']['mean'][error_sweep_idx] = mean(sum_rate_per_monte_carlo)
        metrics['sum_rate']['mmse']['std'][error_sweep_idx] = std(sum_rate_per_monte_carlo)

    print(metrics['sum_rate']['mmse']['mean'])
    print(datetime.now()-real_time_start)

    if config.profile:
        profiler.disable()
        profiler.print_stats(sort='cumulative')

    save_results()

    plot_sweep(csit_error_sweep_range, metrics['sum_rate']['mmse']['mean'],
               'Error', 'Mean Sum Rate', yerr=metrics['sum_rate']['mmse']['std'])

    if config.show_plots:
        plt.show()


if __name__ == '__main__':

    cfg = Config()
    cfg.config_learner.training_name = f'sat_{cfg.sat_nr}_ant_{cfg.sat_tot_ant_nr}_usr_{cfg.user_nr}_satdist_{cfg.sat_dist_average}_usrdist_{cfg.user_dist_average}'

    iterations: int = 10_000
    sweep_range = arange(0.0, 0.6, 0.1)

    test_mmse_precoder_error_sweep(
        config=cfg,
        csit_error_sweep_range=sweep_range,
        monte_carlo_iterations=iterations,
    )


from numpy import (
    arange,
    zeros,
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
from matplotlib.pyplot import (
    show as plt_show,
)

from src.config.config import (
    Config,
)
from src.data.satellite_manager import (
    SatelliteManager,
)
from src.data.user_manager import (
    UserManager,
)
from src.data.precoder.mrc_precoder import (
    mrc_precoder_normalized,
)
from src.data.calc_sum_rate_no_iui import (
    calc_sum_rate_no_iui,
)
from src.data.channel.los_channel_error_model_no_error import (
    los_channel_error_model_no_error,
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


def test_mrc_precoder_distance_sweep(
        config,
        distance_sweep_range,
) -> None:

    def progress_print() -> None:
        progress = (distance_sweep_idx + 1) / (len(distance_sweep_range))
        progress_printer(progress=progress, real_time_start=real_time_start)

    def sim_update():
        user_manager.update_positions(config=config)
        satellite_manager.update_positions(config=config)

        satellite_manager.calculate_satellite_distances_to_users(users=user_manager.users)
        satellite_manager.calculate_satellite_aods_to_users(users=user_manager.users)
        satellite_manager.calculate_steering_vectors_to_users(users=user_manager.users)
        satellite_manager.update_channel_state_information(channel_model=config.channel_model, users=user_manager.users)
        satellite_manager.update_erroneous_channel_state_information(error_model_config=config.error_model, users=user_manager.users)

    def save_results():
        name = f'testing_mrc_sweep_{distance_sweep_range[0]}_{distance_sweep_range[-1]}.gzip'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'distance_sweep')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip_open(Path(results_path, name), 'wb') as file:
            pickle_dump([distance_sweep_range, metrics], file=file)

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)

    real_time_start = datetime.now()

    profiler = None
    if config.profile:
        profiler = start_profiling()

    metrics = {
        'sum_rate': {
            'mrc': {
                'mean': zeros(len(distance_sweep_range)),
                'std': zeros(len(distance_sweep_range)),
            },
        },
    }

    for distance_sweep_idx, distance_sweep_value in enumerate(distance_sweep_range):

        config.user_dist_average = distance_sweep_value
        config.user_dist_bound = 0
        config.error_model.error_model = los_channel_error_model_no_error
        config.error_model.update()

        sim_update()

        w_mrc = mrc_precoder_normalized(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            **config.mrc_args,
        )
        sum_rate = calc_sum_rate_no_iui(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_mrc,
            noise_power_watt=config.noise_power_watt
        )

        metrics['sum_rate']['mrc']['mean'][distance_sweep_idx] = sum_rate
        metrics['sum_rate']['mrc']['std'][distance_sweep_idx] = 0

        if distance_sweep_idx % 10 == 0:
            progress_print()

    if profiler is not None:
        end_profiling(profiler)

    save_results()

    plot_sweep(distance_sweep_range, metrics['sum_rate']['mrc']['mean'],
               'User_dist', 'Mean Sum Rate', yerr=metrics['sum_rate']['mrc']['std'])

    if config.show_plots:
        plt_show()


if __name__ == '__main__':

    cfg = Config()
    cfg.config_learner.training_name = f'sat_{cfg.sat_nr}_ant_{cfg.sat_tot_ant_nr}_usr_{cfg.user_nr}_satdist_{cfg.sat_dist_average}_usrdist_{cfg.user_dist_average}'

    sweep_range = arange(1000-30, 1000+30, 0.01)

    test_mrc_precoder_distance_sweep(
        config=cfg,
        distance_sweep_range=sweep_range,
    )

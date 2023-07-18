

from numpy import (
    newaxis,
    arange,
    zeros,
)
from keras.models import (
    load_model,
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
from src.data.channel.los_channel_error_model_no_error import (
    los_channel_error_model_no_error,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)
from src.utils.real_complex_vector_reshaping import (
    real_vector_to_half_complex_vector,
    complex_vector_to_rad_and_phase,
)
from src.utils.norm_precoder import (
    norm_precoder,
)
from src.utils.plot_sweep import (
    plot_sweep,
)
from src.utils.profiling import (
    start_profiling,
    end_profiling,
)


def test_sac_precoder_distance_sweep(
        config,
        model_parent_path,
        model_name,
        distance_sweep_range,
) -> None:

    def progress_print() -> None:
        progress = (distance_sweep_idx + 1) / (len(distance_sweep_range))
        timedelta = datetime.now() - real_time_start
        finish_time = real_time_start + timedelta / progress

        print(f'\rSimulation completed: {progress:.2%}, '
              f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}', end='')

    def sim_update():
        user_manager.update_positions(config=config)
        satellite_manager.update_positions(config=config)

        satellite_manager.calculate_satellite_distances_to_users(users=user_manager.users)
        satellite_manager.calculate_satellite_aods_to_users(users=user_manager.users)
        satellite_manager.calculate_steering_vectors_to_users(users=user_manager.users)
        satellite_manager.update_channel_state_information(channel_model=config.channel_model, users=user_manager.users)
        satellite_manager.update_erroneous_channel_state_information(error_model_config=config.error_model, users=user_manager.users)

    def save_results():
        name = f'testing_sac_{model_name}_sweep_{distance_sweep_range[0]}_{distance_sweep_range[-1]}.gzip'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'distance_sweep')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip_open(Path(results_path, name), 'wb') as file:
            pickle_dump([distance_sweep_range, metrics], file=file)

    def get_learned_precoder():
        state = config.config_learner.get_state(satellites=satellite_manager, **config.config_learner.get_state_args)
        w_precoder, _ = precoder_network.call(state.astype('float32')[newaxis])
        w_precoder = w_precoder.numpy().flatten()

        # reshape to fit reward calculation
        w_precoder = real_vector_to_half_complex_vector(w_precoder)
        w_precoder = w_precoder.reshape((config.sat_nr * config.sat_ant_nr, config.user_nr))

        # normalize
        return norm_precoder(
            precoding_matrix=w_precoder,
            power_constraint_watt=config.power_constraint_watt,
            per_satellite=True,
            sat_nr=config.sat_nr,
            sat_ant_nr=config.sat_ant_nr)

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)

    network_path = Path(model_parent_path, model_name, 'model')
    precoder_network = load_model(network_path)

    real_time_start = datetime.now()

    profiler = None
    if config.profile:
        profiler = start_profiling()

    metrics = {
        'sum_rate': {
            'mmse': {
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

        precoder_learned = get_learned_precoder()

        sum_rate = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=precoder_learned,
            noise_power_watt=config.noise_power_watt
        )

        metrics['sum_rate']['mmse']['mean'][distance_sweep_idx] = sum_rate
        metrics['sum_rate']['mmse']['std'][distance_sweep_idx] = 0

        if distance_sweep_idx % 10 == 0:
            progress_print()

    if profiler is not None:
        end_profiling(profiler)

    save_results()

    plot_sweep(distance_sweep_range, metrics['sum_rate']['mmse']['mean'],
               'User_dist', 'Mean Sum Rate', yerr=metrics['sum_rate']['mmse']['std'])

    if config.show_plots:
        plt_show()


if __name__ == '__main__':

    cfg = Config()
    cfg.config_learner.training_name = f'sat_{cfg.sat_nr}_ant_{cfg.sat_tot_ant_nr}_usr_{cfg.user_nr}_satdist_{cfg.sat_dist_average}_usrdist_{cfg.user_dist_average}'

    sweep_range = arange(1000-30, 1000+30, 0.01)

    model_path = Path(cfg.trained_models_path,
                      'test',
                      'err_mult_on_steering_cos',
                      'single_error')
    model = 'error_0.0_userwiggle_30_snap_2.974'

    test_sac_precoder_distance_sweep(
        config=cfg,
        distance_sweep_range=sweep_range,
        model_parent_path=model_path,
        model_name=model,
    )

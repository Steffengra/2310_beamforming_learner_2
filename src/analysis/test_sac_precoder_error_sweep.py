
from numpy import (
    newaxis,
    arange,
    zeros,
    mean,
    std,
)
from datetime import (
    datetime,
)
from keras.models import (
    load_model,
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
from src.utils.progress_printer import (
    progress_printer,
)


def test_sac_precoder_error_sweep(
        config,
        model_parent_path,
        model_name,
        csit_error_sweep_range,
        monte_carlo_iterations,
) -> None:

    def progress_print() -> None:
        progress = (error_sweep_idx * monte_carlo_iterations + iter_idx + 1) / (
                len(csit_error_sweep_range) * monte_carlo_iterations)
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

    def sim_update():
        user_manager.update_positions(config=config)
        satellite_manager.update_positions(config=config)

        satellite_manager.calculate_satellite_distances_to_users(users=user_manager.users)
        satellite_manager.calculate_satellite_aods_to_users(users=user_manager.users)
        satellite_manager.calculate_steering_vectors_to_users(users=user_manager.users)
        satellite_manager.update_channel_state_information(channel_model=config.channel_model, users=user_manager.users)
        satellite_manager.update_erroneous_channel_state_information(error_model_config=config.error_model, users=user_manager.users)

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

    def save_results() -> None:
        name = f'testing_sac_{model_name}_sweep_{csit_error_sweep_range[0]}_{csit_error_sweep_range[-1]}_userwiggle_{config.user_dist_bound}.gzip'
        results_path = Path(config.output_metrics_path,
                            config.config_learner.training_name,
                            config.error_model.error_model_name,
                            'error_sweep')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip_open(Path(results_path, name), 'wb') as file:
            pickle_dump([csit_error_sweep_range, metrics], file=file)

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
            'learned': {
                'mean': zeros(len(csit_error_sweep_range)),
                'std': zeros(len(csit_error_sweep_range)),
            },
        },
    }

    for error_sweep_idx, error_sweep_value in enumerate(csit_error_sweep_range):

        # set new error value
        set_new_error_value()

        # set up per monte carlo metrics
        sum_rate_per_monte_carlo = zeros(monte_carlo_iterations)

        for iter_idx in range(monte_carlo_iterations):

            sim_update()

            precoder_learned = get_learned_precoder()

            # get sum rate
            sum_rate = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=precoder_learned,
                noise_power_watt=config.noise_power_watt)

            # log results
            sum_rate_per_monte_carlo[iter_idx] = sum_rate

            if iter_idx % 50 == 0:
                progress_print()

        # log results
        metrics['sum_rate']['learned']['mean'][error_sweep_idx] = mean(sum_rate_per_monte_carlo)
        metrics['sum_rate']['learned']['std'][error_sweep_idx] = std(sum_rate_per_monte_carlo)

    # finish profiling
    if profiler is not None:
        end_profiling(profiler)

    # save results
    save_results()

    plot_sweep(
        x=csit_error_sweep_range,
        y=metrics['sum_rate']['learned']['mean'],
        yerr=metrics['sum_rate']['learned']['std'],
        xlabel='error value',
        ylabel='sum rate',
        title=model_name,
    )

    if config.show_plots:
        plt_show()


if __name__ == '__main__':

    cfg = Config()
    cfg.config_learner.training_name = f'sat_{cfg.sat_nr}_ant_{cfg.sat_tot_ant_nr}_usr_{cfg.user_nr}_satdist_{cfg.sat_dist_average}_usrdist_{cfg.user_dist_average}'

    iterations: int = 10_000
    # sweep_range = arange(0.0, 0.6, 0.1)
    # sweep_range = arange(0.0, 1/10_000_000, 1/100_000_000)
    sweep_range = arange(0, 0.07, 0.005)
    model_path = Path(cfg.trained_models_path,
                      'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000',
                      'err_satpos_and_userpos',
                      'single_error')
    model = 'error_st_0.1_ph_0.01_userwiggle_30_snap_2.785'

    test_sac_precoder_error_sweep(
        config=cfg,
        model_name=model,
        model_parent_path=model_path,
        csit_error_sweep_range=sweep_range,
        monte_carlo_iterations=iterations,
    )

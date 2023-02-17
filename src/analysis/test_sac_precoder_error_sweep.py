
from numpy import (
    arange,
    zeros,
    mean,
    std,
)
from keras.models import (
    load_model,
)
from pathlib import (
    Path,
)
from datetime import (
    datetime,
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
from src.data.satellites import (
    Satellites,
)
from src.data.users import (
    Users,
)
from src.data.channel.los_channel_model import (
    los_channel_model,
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


def test_sac_precoder_error_sweep(
        config,
        model_parent_path,
        model_name,
        csit_error_sweep_range,
        monte_carlo_iterations,
):

    def progress_print() -> None:
        progress = (error_sweep_idx * monte_carlo_iterations + iter_idx + 1) / (
                    len(csit_error_sweep_range) * monte_carlo_iterations)
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

    def get_learned_precoder():
        state = config.config_learner.get_state(satellites=satellites, **config.config_learner.get_state_args)
        w_precoder, _ = precoder_network.get_action_and_log_prob_density(state)
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

    def save_results():
        name = f'testing_sac_{model_name}_sweep_{csit_error_sweep_range[0]}_{csit_error_sweep_range[-1]}_userwiggle_{config.user_dist_variance}.gzip'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'error_sweep')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip_open(Path(results_path, name), 'wb') as file:
            pickle_dump([csit_error_sweep_range, metrics], file=file)

    config = Config()
    satellites = Satellites(config=config)
    users = Users(config=config)

    network_path = Path(model_parent_path, model_name, 'model')
    precoder_network = load_model(network_path)

    real_time_start = datetime.now()
    if config.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

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
        config.error_model.uniform_error_interval['low'] = -1 * error_sweep_value
        config.error_model.uniform_error_interval['high'] = error_sweep_value

        # set up per monte carlo metrics
        sum_rate_per_monte_carlo = {
            'learned': zeros(monte_carlo_iterations),
        }

        for iter_idx in range(monte_carlo_iterations):

            sim_update()

            precoder_learned = get_learned_precoder()

            # get sum rate
            sum_rate_learned = calc_sum_rate(
                channel_state=satellites.channel_state_information,
                w_precoder=precoder_learned,
                noise_power_watt=config.noise_power_watt)

            # log results
            sum_rate_per_monte_carlo['learned'][iter_idx] = sum_rate_learned

            if iter_idx % 50 == 0:
                progress_print()

        # log results
        metrics['sum_rate']['learned']['mean'][error_sweep_idx] = mean(sum_rate_per_monte_carlo['learned'])
        metrics['sum_rate']['learned']['std'][error_sweep_idx] = std(sum_rate_per_monte_carlo['learned'])

    if config.profile:
        profiler.disable()
        profiler.print_stats(sort='cumulative')

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
    sweep_range = arange(0.0, 0.6, 0.1)
    model_path = Path(cfg.trained_models_path, 'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'single_error')
    model = 'error_0.1_userwiggle_100_snapshot_3.303'

    test_sac_precoder_error_sweep(
        config=cfg,
        model_name=model,
        model_parent_path=model_path,
        csit_error_sweep_range=sweep_range,
        monte_carlo_iterations=iterations,
    )

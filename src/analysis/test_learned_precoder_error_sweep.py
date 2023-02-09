

from numpy import (
    newaxis,
    arange,
    zeros,
    mean,
    trace,
    sqrt,
    matmul,
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
    mmse_precoder,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)
from src.models.get_state import (
    get_state_erroneous_channel_state_information,
    get_state_aods,
)
from src.utils.real_complex_vector_reshaping import (
    real_vector_to_half_complex_vector,
    complex_vector_to_double_real_vector,
)

# TODO: move this to proper place
import matplotlib.pyplot as plt


monte_carlo_iterations: int = 10_000


def main():

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

    config = Config()
    satellites = Satellites(config=config)
    users = Users(config=config)

    real_time_start = datetime.now()
    if config.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    sim_update()

    precoder_network = load_model(Path(config.project_root_path, 'models', 'single_error', 'test', 'error_0.1'))

    csit_error_sweep_range = arange(0.0, 0.6, 0.1)
    mean_sum_rate_per_error_value = {
        'learned': zeros(len(csit_error_sweep_range)),
        'mmse': zeros(len(csit_error_sweep_range)),
    }

    for error_sweep_idx, error_sweep_value in enumerate(csit_error_sweep_range):
        config.error_model.uniform_error_interval['low'] = -1 * error_sweep_value
        config.error_model.uniform_error_interval['high'] = error_sweep_value

        sum_rate_per_monte_carlo = {
            'learned': zeros(monte_carlo_iterations),
            'mmse': zeros(monte_carlo_iterations),
        }
        for iter_idx in range(monte_carlo_iterations):

            sim_update()

            state = get_state_aods(satellites=satellites)
            # state = complex_vector_to_double_real_vector(state)
            w_precoder = precoder_network(state[newaxis]).numpy().flatten()
            # print(state)

            # reshape to fit reward calculation
            w_precoder = real_vector_to_half_complex_vector(w_precoder)
            w_precoder = w_precoder.reshape((config.sat_nr*config.sat_ant_nr, config.user_nr))

            # normalize
            norm_factor = sqrt(config.power_constraint_watt / trace(matmul(w_precoder.conj().T, w_precoder)))
            w_precoder_normalized = norm_factor * w_precoder

            # mmse baseline
            w_mmse = mmse_precoder(
                channel_matrix=satellites.erroneous_channel_state_information,
                power_constraint_watt=config.power_constraint_watt,
                noise_power_watt=config.noise_power_watt,
                sat_nr=config.sat_nr,
                sat_ant_nr=config.sat_ant_nr,
            )

            # get sum rate
            sum_rate_learned = calc_sum_rate(
                channel_state=satellites.channel_state_information,
                w_precoder=w_precoder_normalized,
                noise_power_watt=config.noise_power_watt
            )
            sum_rate_per_monte_carlo['learned'][iter_idx] = sum_rate_learned

            sum_rate_mmse = calc_sum_rate(
                channel_state=satellites.channel_state_information,
                w_precoder=w_mmse,
                noise_power_watt=config.noise_power_watt
            )
            sum_rate_per_monte_carlo['mmse'][iter_idx] = sum_rate_mmse

            if iter_idx % 50 == 0:
                progress_print()

        mean_sum_rate_per_error_value['learned'][error_sweep_idx] = mean(sum_rate_per_monte_carlo['learned'])
        mean_sum_rate_per_error_value['mmse'][error_sweep_idx] = mean(sum_rate_per_monte_carlo['mmse'])

    print(mean_sum_rate_per_error_value)
    print(datetime.now() - real_time_start)

    if config.profile:
        profiler.disable()
        profiler.print_stats(sort='cumulative')

    fig, ax = plt.subplots()
    ax.scatter(csit_error_sweep_range, mean_sum_rate_per_error_value['learned'])
    ax.scatter(csit_error_sweep_range, mean_sum_rate_per_error_value['mmse'])
    ax.grid()
    ax.legend(['learned', 'mmse'])

    if config.show_plots:
        plt.show()


if __name__ == '__main__':
    main()

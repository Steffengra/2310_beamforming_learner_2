
from numpy import (
    arange,
    zeros,
    mean,
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


def main():
    def progress_print() -> None:
        progress = (error_sweep_idx * config.monte_carlo_iterations + iter_idx + 1) / (len(csit_error_sweep_range) * config.monte_carlo_iterations)
        timedelta = datetime.now() - real_time_start
        finish_time = real_time_start + timedelta / progress

        print(f'\rSimulation completed: {progress:.2%}, '
              f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}', end='')

    config = Config()
    satellites = Satellites(config=config)
    users = Users(config=config)

    real_time_start = datetime.now()
    if config.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    satellites.calculate_satellite_distances_to_users(users=users.users)
    satellites.calculate_satellite_aods_to_users(users=users.users)
    satellites.calculate_steering_vectors_to_users(users=users.users)

    satellites.update_channel_state_information(channel_model=los_channel_model, users=users.users)

    csit_error_sweep_range = arange(0, 1, 0.1)
    mean_sum_rate_per_error_value = zeros(len(csit_error_sweep_range))

    for error_sweep_idx, error_sweep_value in enumerate(csit_error_sweep_range):
        config.error_model.uniform_error_interval['low'] = -1 * error_sweep_value
        config.error_model.uniform_error_interval['high'] = error_sweep_value

        sum_rate_per_monte_carlo = zeros(config.monte_carlo_iterations)
        for iter_idx in range(config.monte_carlo_iterations):
            satellites.update_erroneous_channel_state_information(
                error_model_config=config.error_model,
                users=users.users
            )

            w_mmse = mmse_precoder(
                channel_matrix=satellites.erroneous_channel_state_information,
                power_constraint_watt=config.power_constraint_watt,
                noise_power_watt=config.noise_power_watt
            )
            sum_rate = calc_sum_rate(
                channel_state=satellites.channel_state_information,
                w_precoder=w_mmse,
                noise_power_watt=config.noise_power_watt
            )
            sum_rate_per_monte_carlo[iter_idx] = sum_rate

            if iter_idx % 50 == 0:
                progress_print()

        mean_sum_rate_per_error_value[error_sweep_idx] = mean(sum_rate_per_monte_carlo)

    print(mean_sum_rate_per_error_value)
    if config.profile:
        profiler.disable()
        profiler.print_stats(sort='cumulative')


if __name__ == '__main__':
    main()

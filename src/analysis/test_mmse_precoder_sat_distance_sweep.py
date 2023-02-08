
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

# TODO: move this to proper place
import matplotlib.pyplot as plt

user_distance_sweep_range = arange(33, 35, 0.001) * 10 ** 3


def main():

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

    sum_rate_per_distance = zeros(len(user_distance_sweep_range))

    for distance_sweep_idx, distance_sweep_value in enumerate(user_distance_sweep_range):
        config.sat_dist_average = distance_sweep_value
        # print('\n', config.sat_dist_average, '\n')

        sim_update()

        w_mmse = mmse_precoder(
            channel_matrix=satellites.erroneous_channel_state_information,
            power_constraint_watt=config.power_constraint_watt,
            noise_power_watt=config.noise_power_watt,
            sat_nr=config.sat_nr,
            sat_ant_nr=config.sat_ant_nr,
        )
        sum_rate = calc_sum_rate(
            channel_state=satellites.channel_state_information,
            w_precoder=w_mmse,
            noise_power_watt=config.noise_power_watt
        )

        sum_rate_per_distance[distance_sweep_idx] = sum_rate

    print(sum_rate_per_distance)
    print(datetime.now()-real_time_start)

    if config.profile:
        profiler.disable()
        profiler.print_stats(sort='cumulative')

    fig, ax = plt.subplots()
    ax.plot(user_distance_sweep_range, sum_rate_per_distance)
    ax.grid()

    if config.show_plots:
        plt.show()


if __name__ == '__main__':
    main()

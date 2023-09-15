
import gzip
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.models import load_model
from matplotlib.pyplot import show as plt_show

from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.config.config import Config
from src.data.precoder.robust_SLNR_precoder import robust_SLNR_precoder_no_norm
from src.data.precoder.calc_autocorrelation import calc_autocorrelation
from src.data.precoder.mmse_precoder import mmse_precoder_normalized, mmse_precoder_no_norm
from src.data.calc_sum_rate import calc_sum_rate
from src.utils.plot_beampattern import plot_beampattern
from src.utils.update_sim import update_sim
from src.utils.real_complex_vector_reshaping import real_vector_to_half_complex_vector
from src.utils.norm_precoder import norm_precoder


plot = [
    'mmse',
    # 'slnr',
    'learned',
    # 'ones',
]

angle_sweep_range = np.arange((90 - 30) * np.pi / 180, (90 + 30) * np.pi / 180, 0.1 * np.pi / 180)  # arange or None


config = Config()
# config.user_dist_bound = 0  # disable user wiggle
config.user_dist_bound = 50_000

model_path = Path(  # SAC only
    config.trained_models_path,
    '1_sat_8_ant_3_usr_100000_dist_0.0_error_on_cos_0.1_fading',
    'single_error',
    'userwiggle_50000_snap_4.008',
    'model',
)

if 'learned' in plot:
    from src.utils.compare_configs import compare_configs
    compare_configs(config, Path(model_path, '..', 'config'))

satellite_manager = SatelliteManager(config)
user_manager = UserManager(config)

for iter_id in range(2):

    update_sim(config=config, satellite_manager=satellite_manager, user_manager=user_manager)

    # MMSE
    if 'mmse' in plot:
        w_mmse = mmse_precoder_normalized(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            **config.mmse_args,
        )

        test = mmse_precoder_no_norm(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            noise_power_watt=config.noise_power_watt,
            power_constraint_watt=config.power_constraint_watt,
        )

        sum_rate_mmse = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_mmse,
            noise_power_watt=config.noise_power_watt,
        )

        plot_beampattern(
            satellite=satellite_manager.satellites[0],
            users=user_manager.users,
            w_precoder=w_mmse,
            plot_title='mmse',
            angle_sweep_range=angle_sweep_range,
        )

        print(f'mmse: {sum_rate_mmse}')

    # SLNR
    if 'slnr' in plot:
        autocorrelation = calc_autocorrelation(
            satellite=satellite_manager.satellites[0],
            error_model_config=config.config_error_model,
            error_distribution='uniform',
        )

        w_slnr = robust_SLNR_precoder_no_norm(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            autocorrelation_matrix=autocorrelation,
            noise_power_watt=config.noise_power_watt,
            power_constraint_watt=config.power_constraint_watt,
        )

        sum_rate_slnr = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_slnr,
            noise_power_watt=config.noise_power_watt,
        )

        plot_beampattern(
            satellite=satellite_manager.satellites[0],
            users=user_manager.users,
            w_precoder=w_slnr,
            plot_title='slnr',
            angle_sweep_range=angle_sweep_range,
        )
        print(f'slnr: {sum_rate_slnr}')

    # Learned
    if 'learned' in plot:

        with tf.device('CPU:0'):

            with gzip.open(Path(model_path, '..', 'config', 'norm_dict.gzip')) as file:
                norm_dict = pickle.load(file)
            norm_factors = norm_dict['norm_factors']
            if norm_factors != {}:
                config.config_learner.get_state_args['norm_state'] = True

            print(norm_factors)

            precoder_network = load_model(model_path)

            state = config.config_learner.get_state(
                satellite_manager=satellite_manager,
                norm_factors=norm_factors,
                **config.config_learner.get_state_args
            )
            w_precoder, _ = precoder_network.call(state.astype('float32')[np.newaxis])
            w_precoder = w_precoder.numpy().flatten()

            w_precoder = real_vector_to_half_complex_vector(w_precoder)
            w_precoder = w_precoder.reshape((config.sat_nr * config.sat_ant_nr, config.user_nr))

            w_learned = norm_precoder(
                precoding_matrix=w_precoder,
                power_constraint_watt=config.power_constraint_watt,
                per_satellite=True,
                sat_nr=config.sat_nr,
                sat_ant_nr=config.sat_ant_nr
            )

            sum_rate_learned = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_learned,
                noise_power_watt=config.noise_power_watt,
            )

        print(f'learned: {sum_rate_learned}')

        plot_beampattern(
            satellite=satellite_manager.satellites[0],
            users=user_manager.users,
            w_precoder=w_learned,
            plot_title='learned',
            angle_sweep_range=angle_sweep_range,
        )

    # Ones
    if 'ones' in plot:
        w_ones = np.ones(w_mmse.shape)

        sum_rate_ones = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_ones,
            noise_power_watt=config.noise_power_watt,
        )

        plot_beampattern(
            satellite=satellite_manager.satellites[0],
            users=user_manager.users,
            w_precoder=w_ones,
            plot_title='ones',
            angle_sweep_range=angle_sweep_range,
        )

        print(f'ones: {sum_rate_ones}')

    plt_show()

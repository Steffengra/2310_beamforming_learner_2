
import numpy as np
from copy import (
    deepcopy,
)

from src.config.config import (
    Config,
)
from src.models.train_sac import (
    train_sac_single_error,
)
from src.analysis.helpers.test_sac_precoder_error_sweep import (
    test_sac_precoder_error_sweep,
)
from src.analysis.helpers.test_sac_precoder_user_distance_sweep import (
    test_sac_precoder_user_distance_sweep,
)


def learn_on_userdist_and_mult_error(
        userdist,
        mult_error,
) -> None:

    cfg = Config()
    cfg.profile = False
    cfg.show_plots = False
    cfg.config_learner.training_name = f'sat_{cfg.sat_nr}_ant_{cfg.sat_tot_ant_nr}_usr_{cfg.user_nr}_satdist_{cfg.sat_dist_average}_usrdist_{userdist}'
    cfg.user_dist_average = userdist
    cfg.config_error_model.config_error_model = los_channel_error_model_multiplicative_on_cos
    cfg.config_error_model.update()
    cfg.config_error_model.uniform_error_interval['low'] = -mult_error
    cfg.config_error_model.uniform_error_interval['high'] = mult_error

    best_model_path = train_sac_single_error(config=cfg)

    if test:
        test_precoder(config=cfg,
                      model_path=best_model_path,
                      error_sweep_range=testing_error_sweep_range_mult_on_steering)


def learn_on_userdist_and_sat2userdist_error(
        userdist,
        sat2userdisterror_std,
) -> None:

    cfg = Config()
    cfg.profile = False
    cfg.show_plots = False
    cfg.config_learner.training_name = f'sat_{cfg.sat_nr}_ant_{cfg.sat_tot_ant_nr}_usr_{cfg.user_nr}_satdist_{cfg.sat_dist_average}_usrdist_{userdist}'
    cfg.user_dist_average = userdist
    cfg.config_error_model.config_error_model = los_channel_error_model_in_sat2user_dist
    cfg.config_error_model.update()
    cfg.config_error_model.distance_error_std = sat2userdisterror_std

    best_model_path = train_sac_single_error(config=deepcopy(cfg))

    if test:
        test_precoder(config=deepcopy(cfg),
                      model_path=best_model_path,
                      error_sweep_range=testing_error_sweep_range_sat2userdist)


def learn_on_userdist_and_satpos_and_userpos_error(
        userdist,
        phase_sat_error_std,
        mult_error_bound,
):
    cfg = Config()
    cfg.profile = False
    cfg.show_plots = False
    cfg.config_learner.training_name = f'sat_{cfg.sat_nr}_ant_{cfg.sat_tot_ant_nr}_usr_{cfg.user_nr}_satdist_{cfg.sat_dist_average}_usrdist_{userdist}'
    cfg.user_dist_average = userdist
    cfg.config_error_model.config_error_model = los_channel_error_model_in_sat_and_user_pos
    cfg.config_error_model.update()
    cfg.config_error_model.uniform_error_interval['low'] = -mult_error_bound
    cfg.config_error_model.uniform_error_interval['high'] = mult_error_bound
    cfg.config_error_model.phase_sat_error_std = phase_sat_error_std

    best_model_path = train_sac_single_error(config=cfg)

    if test:
        test_precoder(config=cfg,
                      model_path=best_model_path,
                      error_sweep_range=testing_error_sweep_range_err_satpos_and_userpos)


def test_precoder(
        config,
        model_path,
        error_sweep_range,
):
    model_parent_path = model_path.parent
    model_name = model_path.name
    monte_carlo_iterations = 10_000

    # distance within wiggle
    test_sac_precoder_user_distance_sweep(
        config=deepcopy(config),
        model_path=model_path,
        distance_sweep_range=np.arange(config.user_dist_average - config.user_dist_bound,
                                       config.user_dist_average + config.user_dist_bound,
                                       0.01),
    )

    # distance outside wiggle
    test_sac_precoder_user_distance_sweep(
        config=deepcopy(config),
        model_path=model_path,
        distance_sweep_range=np.arange(config.user_dist_average - 10 * config.user_dist_bound,
                                       config.user_dist_average + 10 * config.user_dist_bound,
                                       0.01),
    )

    # error with wiggle
    test_sac_precoder_error_sweep(
        config=deepcopy(config),
        model_parent_path=model_parent_path,
        model_name=model_name,
        error_sweep_range=error_sweep_range,
        monte_carlo_iterations=monte_carlo_iterations,
    )

    # error no wiggle
    old_user_dist_bound = config.user_dist_bound
    config.user_dist_bound = 0.0
    test_sac_precoder_error_sweep(
        config=deepcopy(config),
        model_parent_path=model_parent_path,
        model_name=model_name,
        error_sweep_range=error_sweep_range,
        monte_carlo_iterations=monte_carlo_iterations,
    )
    config.user_dist_bound = old_user_dist_bound


test = True
testing_error_sweep_range_sat2userdist = np.arange(0.0, 1 / 10_000_000, 1 / 100_000_000)
testing_error_sweep_range_mult_on_steering = np.arange(0.0, 0.6, 0.1)
testing_error_sweep_range_err_satpos_and_userpos = np.arange(0.0, 0.1, 0.01)


def main():

    # learn_on_userdist_and_mult_error(userdist=1000, mult_error=0.0)
    # learn_on_userdist_and_mult_error(userdist=1000, mult_error=0.1)
    # learn_on_userdist_and_mult_error(userdist=1000, mult_error=0.2)
    # learn_on_userdist_and_mult_error(userdist=1000, mult_error=0.3)
    # learn_on_userdist_and_mult_error(userdist=1000, mult_error=0.4)
    # learn_on_userdist_and_mult_error(userdist=1000, mult_error=0.5)

    # learn_on_userdist_and_mult_error(userdist=50_000, mult_error=0.0)
    # learn_on_userdist_and_mult_error(userdist=50_000, mult_error=0.1)
    # learn_on_userdist_and_mult_error(userdist=50_000, mult_error=0.2)
    # learn_on_userdist_and_mult_error(userdist=50_000, mult_error=0.3)
    # learn_on_userdist_and_mult_error(userdist=50_000, mult_error=0.4)
    # learn_on_userdist_and_mult_error(userdist=50_000, mult_error=0.5)

    # learn_on_userdist_and_satpos_and_userpos_error(userdist=1_000, phase_sat_error_std=0.0, mult_error_bound=0.0)
    learn_on_userdist_and_satpos_and_userpos_error(userdist=1_000, phase_sat_error_std=0.01, mult_error_bound=0.1)

    # learn_on_userdist_and_mult_error(userdist=100_000, mult_error=0.0)
    # learn_on_userdist_and_mult_error(userdist=100_000, mult_error=0.1)
    # learn_on_userdist_and_mult_error(userdist=100_000, mult_error=0.)
    # learn_on_userdist_and_mult_error(userdist=100_000, mult_error=0.3)
    # learn_on_userdist_and_mult_error(userdist=100_000, mult_error=0.4)
    # learn_on_userdist_and_mult_error(userdist=100_000, mult_error=0.5)

    # learn_on_userdist_and_sat2userdist_error(userdist=1000, sat2userdisterror_std=0)
    # learn_on_userdist_and_sat2userdist_error(userdist=1000, sat2userdisterror_std=1/100_000_000)
    # learn_on_userdist_and_sat2userdist_error(userdist=1000, sat2userdisterror_std=2/100_000_000)
    # learn_on_userdist_and_sat2userdist_error(userdist=1000, sat2userdisterror_std=3/100_000_000)
    # learn_on_userdist_and_sat2userdist_error(userdist=1000, sat2userdisterror_std=4/100_000_000)
    # learn_on_userdist_and_sat2userdist_error(userdist=1000, sat2userdisterror_std=5/100_000_000)
    # learn_on_userdist_and_sat2userdist_error(userdist=1000, sat2userdisterror_std=6/100_000_000)
    # learn_on_userdist_and_sat2userdist_error(userdist=1000, sat2userdisterror_std=7/100_000_000)
    # learn_on_userdist_and_sat2userdist_error(userdist=1000, sat2userdisterror_std=8/100_000_000)
    # learn_on_userdist_and_sat2userdist_error(userdist=1000, sat2userdisterror_std=9/100_000_000)


if __name__ == '__main__':
    main()

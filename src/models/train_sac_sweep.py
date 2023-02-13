
from src.config.config import (
    Config,
)
from src.models.train_sac import (
    train_sac_single_error,
)


def main():

    def learn_on_userdist_and_mult_error(
            userdist,
            mult_error,
    ) -> None:

        cfg = Config()
        cfg.profile = False
        cfg.show_plots = False
        cfg.config_learner.training_name = f'sat_{cfg.sat_nr}_ant_{cfg.sat_tot_ant_nr}_usr_{cfg.user_nr}_usrdist_{userdist}_mult_error_{mult_error}'
        cfg.user_dist_average = userdist
        cfg.error_model.uniform_error_interval['low'] = -mult_error
        cfg.error_model.uniform_error_interval['high'] = mult_error

        train_sac_single_error(config=cfg)

    learn_on_userdist_and_mult_error(userdist=1000, mult_error=0.0)
    learn_on_userdist_and_mult_error(userdist=1000, mult_error=0.1)
    learn_on_userdist_and_mult_error(userdist=1000, mult_error=0.2)
    learn_on_userdist_and_mult_error(userdist=1000, mult_error=0.3)
    learn_on_userdist_and_mult_error(userdist=1000, mult_error=0.4)
    learn_on_userdist_and_mult_error(userdist=1000, mult_error=0.5)

    learn_on_userdist_and_mult_error(userdist=10_000, mult_error=0.0)
    learn_on_userdist_and_mult_error(userdist=10_000, mult_error=0.1)
    learn_on_userdist_and_mult_error(userdist=10_000, mult_error=0.2)
    learn_on_userdist_and_mult_error(userdist=10_000, mult_error=0.3)
    learn_on_userdist_and_mult_error(userdist=10_000, mult_error=0.4)
    learn_on_userdist_and_mult_error(userdist=10_000, mult_error=0.5)


if __name__ == '__main__':
    main()

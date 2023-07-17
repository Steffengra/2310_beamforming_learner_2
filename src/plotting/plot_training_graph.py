
import matplotlib.pyplot as plt
from gzip import (
    open as gzip_open,
)
from pickle import (
    load as pickle_load,
)
from pathlib import (
    Path,
)

from src.config.config import (
    Config,
)
from src.config.config_plotting import (
    PlotConfig,
    generic_styling,
)


def plot_training_graph(
        path,
        name,
        xlabel: None or str = None,
        ylabel: None or str = None,
):

    plot_config = PlotConfig()

    width = plot_config.textwidth
    height = width * 2 / 3

    with gzip_open(path, 'rb') as file:
        data = pickle_load(file)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.scatter(range(len(data['mean_sum_rate_per_episode'])), data['mean_sum_rate_per_episode'])

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)

    plot_config.save_figures(plot_name=name, padding=0)


if __name__ == '__main__':
    cfg = Config()
    path = Path(cfg.output_metrics_path, 'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'err_mult_on_steering_cos', 'single_error', 'training_error_0.0_userwiggle_30.gzip')

    plot_training_graph(path, name='training_test', xlabel='Training Episode', ylabel='Mean Reward')
    plt.show()

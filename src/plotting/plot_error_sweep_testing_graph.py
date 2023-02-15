
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


def plot_error_sweep_testing_graph(
        paths,
        name,
        legend: list or None = None,
) -> None:

    plot_config = PlotConfig()

    width = plot_config.textwidth
    height = width * 2 / 3

    fig, ax = plt.subplots(figsize=(width, height))

    if type(paths) == list:
        data = []
        for path in paths:
            with gzip_open(path, 'rb') as file:
                data.append(pickle_load(file))
        for data_entry in data:
            first_entry = list(data_entry[1]['sum_rate'].keys())[0]
            ax.errorbar(data_entry[0],
                        data_entry[1]['sum_rate'][first_entry]['mean'],
                        yerr=data_entry[1]['sum_rate'][first_entry]['std'])
    else:
        with gzip_open(paths, 'rb') as file:
            data = (pickle_load(file))

            first_entry = list(data[1]['sum_rate'].keys())[0]
            ax.errorbar(data[0],
                        data[1]['sum_rate'][first_entry]['mean'],
                        yerr=data[1]['sum_rate'][first_entry]['std'])

    ax.set_xlabel('Error Value')
    ax.set_ylabel('Reward')

    if legend:
        ax.legend(legend)

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)

    plot_config.save_figures(plot_name=name, padding=0)


if __name__ == '__main__':
    cfg = Config()
    data_paths = [
        Path(cfg.output_metrics_path,
             'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'error_sweep',
             'testing_mmse_sweep_0.0_0.5_userwiggle_100.gzip'),

        Path(cfg.output_metrics_path,
             'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'error_sweep',
             'testing_mmse_sweep_0.0_0.5_userwiggle_100.gzip'),
    ]
    plot_legend = ['mmse1', 'mmse2']

    plot_error_sweep_testing_graph(paths=data_paths, name='testing_error_sweep', legend=plot_legend)
    plt.show()

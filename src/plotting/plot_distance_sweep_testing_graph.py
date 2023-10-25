
import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np
from pathlib import (
    Path,
)

from src.config.config import (
    Config,
)
from src.config.config_plotting import (
    PlotConfig,
    save_figures,
    generic_styling,
)


def plot_distance_sweep_testing_graph(
        paths,
        name,
        width,
        height,
        plots_parent_path,
        legend: list or None = None,
        colors: list or None = None,
        markerstyle: list or None = None,
        linestyles: list or None = None,
) -> None:

    fig, ax = plt.subplots(figsize=(width, height))

    data = []
    for path in paths:
        with gzip.open(path, 'rb') as file:
            data.append(pickle.load(file))
    for data_id, data_entry in enumerate(data):
        first_entry = list(data_entry[1]['sum_rate'].keys())[0]
        if markerstyle is not None:
            marker = markerstyle[data_id]
        else:
            marker = None

        if colors is not None:
            color = colors[data_id]
        else:
            color = None

        if linestyles is not None:
            linestyle = linestyles[data_id]
        else:
            linestyle = None

        if len(data_entry[0][data_entry[1]['sum_rate']['mean'] > 0.999 * np.max(data_entry[1]['sum_rate']['mean'])]) < int(len(data_entry[0])/10):
            markevery = np.searchsorted(data_entry[0], data_entry[0][data_entry[1]['sum_rate']['mean'] > 0.999 * np.max(data_entry[1]['sum_rate']['mean'])])
            for value_id in reversed(range(1, len(markevery))):
                if markevery[value_id] / markevery[value_id-1] < 1.01:
                    markevery = np.delete(markevery, value_id)
        else:
            markevery = (int(len(data_entry[0])/10 / len(data)*data_id), int(len(data_entry[0])/10))

        ax.plot(data_entry[0],
                data_entry[1]['sum_rate']['mean'],
                color=color,
                marker=marker,
                markevery=markevery,
                linestyle=linestyle
                )

    ax.set_xlabel('User Distance \( D_{usr} \) [m]')
    ax.set_ylabel('Sum Rate \( R \) [bps/Hz]')

    if legend:
        ax.legend(legend, ncols=3)

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)

    save_figures(plots_parent_path=plots_parent_path, plot_name=name, padding=0)


if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path,
             'sat_1_ant_16_usr_3_satdist_10000_usrdist_100000', 'distance_sweep',
             'testing_mmse_sweep_50000_149999.gzip'),

        Path(cfg.output_metrics_path,
             'sat_1_ant_16_usr_3_satdist_10000_usrdist_100000', 'distance_sweep',
             'testing_learned_0.0_error_sweep_50000_149999.gzip'),
        # Path(cfg.output_metrics_path,
        #      'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'distance_sweep',
        #      'testing_sac_error_0.1_userwiggle_30_snap_3.422_sweep_970.0_1029.9899999999454.gzip'),
        # Path(cfg.output_metrics_path,
        #      'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'distance_sweep',
        #      'testing_mrc_sweep_970.0_1029.9899999999454.gzip'),

    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 1/3

    plot_legend = ['MMSE', 'SAC1', 'OMA']
    plot_markerstyle = ['o', 's', '^']
    plot_colors = [plot_cfg.cp2['blue'], plot_cfg.cp2['magenta'], plot_cfg.cp2['black']]
    plot_linestyles = ['-', '-', '--']

    plot_distance_sweep_testing_graph(
        paths=data_paths,
        name='dist_sweep_test_long',
        width=plot_width,
        height=plot_height,
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    plt.show()

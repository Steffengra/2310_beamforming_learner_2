
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
    save_figures,
    generic_styling,
)


def plot_error_sweep_testing_graph(
        paths,
        name,
        width,
        height,
        xlabel,
        ylabel,
        plots_parent_path,
        legend: list or None = None,
        colors: list or None = None,
        markerstyle: list or None = None,
        linestyles: list or None = None,
) -> None:

    fig, ax = plt.subplots(figsize=(width, height))

    data = []
    for path in paths:
        with gzip_open(path, 'rb') as file:
            data.append(pickle_load(file))

    for data_id, data_entry in enumerate(data):

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

        ax.errorbar(
            data_entry[0],
            data_entry[1]['sum_rate']['mean'],
            yerr=data_entry[1]['sum_rate']['std'],
            marker=marker,
            color=color,
            linestyle=linestyle,
            label=legend[data_id],
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    marker_training_sac1 = plt.plot(0, -0.1, marker='^', color=colors[2], label='_')
    marker_training_sac1[0].set_clip_on(False)
    marker_training_sac2 = plt.plot(0.05, -0.1, marker='^', color=colors[3], label='_')
    marker_training_sac2[0].set_clip_on(False)

    ax.set_ylim([0, 4.5])

    if legend:
        ax.legend(ncols=2)

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)

    save_figures(plots_parent_path=plots_parent_path, plot_name=name, padding=0)


if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path,
             'sat_1_ant_16_usr_3_satdist_10000_usrdist_10000', 'error_sweep',
             'testing_mmse_sweep_0.0_0.1_userwiggle_5000.gzip'),
        Path(cfg.output_metrics_path,
             'sat_1_ant_16_usr_3_satdist_10000_usrdist_10000', 'error_sweep',
             'testing_robust_slnr_sweep_0.0_0.1_userwiggle_5000.gzip'),
        Path(cfg.output_metrics_path,
             'sat_1_ant_16_usr_3_satdist_10000_usrdist_10000', 'error_sweep',
             'testing_learned_0.0_error_sweep_0.0_0.1_userwiggle_5000.gzip'),
        Path(cfg.output_metrics_path,
             'sat_1_ant_16_usr_3_satdist_10000_usrdist_10000', 'error_sweep',
             'testing_learned_0.5_error_sweep_0.0_0.1_userwiggle_5000.gzip'),
    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 9 / 20

    plot_legend = ['MMSE', 'rSLNR', 'pSAC3', 'rSAC3']
    plot_markerstyle = ['o', '^', 's', 'x']
    plot_colors = [plot_cfg.cp2['gold'], plot_cfg.cp2['magenta'], plot_cfg.cp2['blue'], plot_cfg.cp2['green']]
    plot_linestyles = ['-', '--', '-', '-']

    plot_error_sweep_testing_graph(
        paths=data_paths,
        name='error_sweep_16_ant_close_paper',
        width=plot_width,
        xlabel='Error Bound \( apfel \)',
        ylabel='Sum Rate \( B \) [bit/s/Hz]',
        height=plot_height,
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    plt.show()

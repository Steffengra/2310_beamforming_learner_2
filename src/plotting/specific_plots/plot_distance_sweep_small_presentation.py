
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.pyplot import show

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.plotting.plot_distance_sweep_testing_graph import plot_distance_sweep_testing_graph


if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    plt.rc('font', family='sans-serif')  # label fonts, ['serif', >'sans-serif', 'monospace']

    data_paths = [
        Path(cfg.output_metrics_path,
             'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'distance_sweep',
             'testing_mmse_sweep_970.0_1029.9899999999454.gzip'),

        Path(cfg.output_metrics_path,
             'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'distance_sweep',
             'testing_sac_error_0.0_userwiggle_30_snap_3.580_sweep_970.0_1029.9899999999454.gzip'),
        Path(cfg.output_metrics_path,
             'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'distance_sweep',
             'testing_mrc_sweep_970.0_1029.9899999999454.gzip'),

    ]

    plot_width = 0.99 * 2.67
    plot_height = plot_width * 2/3

    plot_legend = ['MMSE', 'SAC1', 'OMA']
    plot_markerstyle = ['o', 's', '^']
    plot_colors = [plot_cfg.cp2['blue'], plot_cfg.cp2['magenta'], plot_cfg.cp2['black']]
    plot_linestyles = ['-', '-', '--']

    plot_distance_sweep_testing_graph(
        paths=data_paths,
        name='distance_sweep_small_presentation',
        width=plot_width,
        height=plot_height,
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    show()

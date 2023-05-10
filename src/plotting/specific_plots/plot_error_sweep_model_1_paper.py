
from pathlib import Path
from matplotlib.pyplot import show

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.plotting.plot_error_sweep_testing_graph import plot_error_sweep_testing_graph


if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path,
             'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'err_mult_on_steering_cos', 'error_sweep',
             'testing_mmse_sweep_0.0_0.5_userwiggle_30.gzip'),
        Path(cfg.output_metrics_path,
             'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'err_mult_on_steering_cos', 'error_sweep',
             'testing_mrc_sweep_0.0_0.5_userwiggle_30.gzip'),

        Path(cfg.output_metrics_path,
             'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'err_mult_on_steering_cos', 'error_sweep',
             'testing_sac_error_0.0_userwiggle_30_snap_3.580_sweep_0.0_0.5_userwiggle_30.gzip'),
        Path(cfg.output_metrics_path,
             'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'err_mult_on_steering_cos', 'error_sweep',
             'testing_sac_error_0.1_userwiggle_30_snap_3.422_sweep_0.0_0.5_userwiggle_30.gzip'),

    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 12 / 30

    plot_legend = ['MMSE', 'OMA', 'SAC1', 'SAC2']
    plot_markerstyle = ['o', '^', 's', 'x']
    plot_colors = [plot_cfg.cp2['blue'], plot_cfg.cp2['black'], plot_cfg.cp2['magenta'], plot_cfg.cp2['green']]
    plot_linestyles = ['-', '--', '-', '-']

    plot_error_sweep_testing_graph(
        paths=data_paths,
        name='error_sweep_model_1_paper',
        width=plot_width,
        height=plot_height,
        xlabel='User Position Error Bound \( \Delta\epsilon \)',
        ylabel='Sum Rate \( \hat{R} \) [bps/Hz]',
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    show()

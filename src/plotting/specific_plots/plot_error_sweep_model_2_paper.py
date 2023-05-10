
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
             'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'err_satpos_and_userpos', 'error_sweep',
             'testing_mmse_sweep_0.0_0.07_userwiggle_30.gzip'),

        Path(cfg.output_metrics_path,
             'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'err_satpos_and_userpos', 'error_sweep',
             'testing_sac_error_st_0.1_ph_0.01_userwiggle_30_snap_2.785_sweep_0.0_0.07_userwiggle_30.gzip'),
    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 12 / 30

    plot_legend = ['MMSE', 'SAC3']
    plot_markerstyle = ['o', 'x']
    plot_colors = [plot_cfg.cp2['blue'], plot_cfg.cp2['gold']]
    plot_linestyles = ['-', '-']

    plot_error_sweep_testing_graph(
        paths=data_paths,
        name='error_sweep_model_2_paper',
        width=plot_width,
        height=plot_height,
        xlabel='Satellite Position Error Scale \( \sigma_{\eta} \)',
        ylabel='Sum Rate \( R \) [bps/Hz]',
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    show()

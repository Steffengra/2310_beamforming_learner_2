
import gzip
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors

from src.config.config import (
    Config,
)
from src.config.config_plotting import (
    PlotConfig,
    save_figures,
    generic_styling,
)


def plot_beam_patterns(
        width,
        height,
        path,
        plots: list,
        color_dict: dict,
        line_style_dict: dict,
        label_dict: dict,
        xlim: list,
) -> None:

    rows = max([subdict['row'] for subdict in plots]) + 1
    cols = max([subdict['column'] for subdict in plots]) + 1

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex='all',
        sharey='all',
        figsize=(width, height),
    )

    with gzip.open(path, 'rb') as file:
        data_file = pickle.load(file)
        data = data_file[1]
        angle_sweep_range = data_file[0]

    for plot in plots:
        row = plot['row']
        col = plot['column']
        realization = plot['realization']
        precoders = plot['precoders']
        print(f'Plot {row}, {col}')

        if rows > 1 and cols > 1:
            ax = axes[row, col]
        elif cols > 1:
            ax = axes[col]
        elif rows > 1:
            ax = axes[row]
        else:
            ax = axes

        # User positions
        for user_id, user_position in enumerate(data[realization]['user_positions']):
            if user_id == 0:
                label = 'Users'
            else:
                label = '_UserHidden'

            ax.scatter(
                user_position,
                0,
                label=label,
                color='black',
            )
            ax.axvline(
                user_position,
                label='_UserHidden',
                # color=mpl_colors.TABLEAU_COLORS[list(mpl_colors.TABLEAU_COLORS.keys())[user_idx]],
                color='black',
                linestyle='dotted',
            )

        # Beam patterns
        for precoder in precoders:
            for user_id in range(data[realization][precoder]['power_gains'].shape[0]):
                if user_id == 0:
                    label = label_dict[precoder]
                else:
                    label = '_PrecoderHidden'

                ax.plot(
                    angle_sweep_range,
                    data[realization][precoder]['power_gains'][user_id],
                    label=label,
                    color=color_dict[precoder],
                    linestyle=line_style_dict[precoder],
                )

            print(f'{precoder} sum rate: {data[realization][precoder]["sum_rate"]}')

        ax.set_xlim(xlim)

        # ax.legend()

        if row == (rows-1):
            ax.set_xlabel('xlabel')
        if col == 0:
            ax.set_ylabel('ylabel')

        generic_styling(ax=ax)

    fig.tight_layout(pad=0)


def print_realizations(
        path,
) -> None:

    with gzip.open(path, 'rb') as file:
        data = pickle.load(file)[1]

    for date_entry_id, data_entry in enumerate(data):
        print(f'{date_entry_id}', '', end='')
        for error in data_entry['estimation_errors']:
            if any(data_entry['estimation_errors'][error]) != 0:
                print(error, data_entry['estimation_errors'][error], '', end='')

        print('sum rate: ', end='')
        for key in data_entry:
            if key not in ['estimation_errors', 'user_positions']:
                print(key, f'{data_entry[key]["sum_rate"]:.2f},', '', end='')
        print('')


if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    list_patterns = False

    which_plots = [
        {
            'row': 0,
            'column': 0,
            'realization': 2,
            'precoders': ['mmse',]
        },
        # {
        #     'row': 0,
        #     'column': 1,
        #     'realization': 2,
        #     'precoders': ['slnr',]
        # },
        # {
        #     'row': 0,
        #     'column': 2,
        #     'realization': 2,
        #     'precoders': ['something',]
        # },
    ]

    data_path = Path(cfg.output_metrics_path,
                     'sat_1_ant_16_usr_3_satdist_10000_usrdist_100000', 'beam_patterns', 'beam_patterns.gzip')

    plot_width = 0.99 * plot_cfg.textwidth
    # plot_width = 0.99 * 3.5
    plot_height = plot_width * 1/2

    x_limits = [1.35, 1.85]
    # x_limits = None

    colors = {
        'mmse': plot_cfg.cp2['blue'],
        'slnr': plot_cfg.cp2['green'],
        'something': plot_cfg.cp2['gold'],
    }

    line_styles = {
        'mmse': 'solid',
        'slnr': 'dashed',
        'something': 'dashed',
    }

    labels = {
        'mmse': 'MMSE',
        'slnr': 'SLNR',
        'something': 'abc',
    }

    if list_patterns:
        print_realizations(data_path)

    plot_beam_patterns(
        width=plot_width,
        height=plot_height,
        path=data_path,
        plots=which_plots,
        color_dict=colors,
        line_style_dict=line_styles,
        label_dict=labels,
        xlim=x_limits,
    )

    plt.show()

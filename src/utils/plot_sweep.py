
import matplotlib.pyplot as plt


def plot_sweep(
        x,
        y,
        xlabel,
        ylabel,
        yerr=(),
        legend=(),
        title='',
):
    fig, ax = plt.subplots()

    if type(y) is list:
        for y_idx, y_category in enumerate(y):
            if yerr:
                ax.errorbar(x, y_category, yerr=yerr[y_idx])
            else:
                ax.scatter(x, y_category)

    else:
        if any(yerr):
            ax.errorbar(x, y, yerr=yerr)
        else:
            ax.scatter(x, y)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if legend:
        ax.legend(legend)

    if title:
        fig.suptitle(title)

    ax.grid()

    fig.tight_layout()

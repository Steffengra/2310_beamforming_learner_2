
import matplotlib.pyplot as plt


def plot_sweep(
        x,
        y,
        xlabel,
        ylabel,
        yerr=(),
):
    fig, ax = plt.subplots()
    if any(yerr):
        ax.errorbar(x, y, yerr=yerr)
    else:
        ax.scatter(x, y)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid()

    fig.tight_layout()

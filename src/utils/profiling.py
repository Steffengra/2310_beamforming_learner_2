
"""Provides functions for quick profiling."""


def start_profiling():
    """Import cprofile, create profiler and start profiling."""

    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    return profiler


def end_profiling(profiler):
    """Close profiling and print results."""

    profiler.disable()
    profiler.print_stats(sort='cumulative')

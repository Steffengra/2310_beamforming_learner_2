
def start_profiling():

    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    return profiler


def end_profiling(profiler):

    profiler.disable()
    profiler.print_stats(sort='cumulative')

"""
Pipeline (core): the orchestration used by the notebooks and the command line: G(x) and
J(x) for a set of images (`per_image`), the comparison between configurations
(`alignment`), the local metric field (`metric_field`), the estimates at each noise
scale for a set of example images (`per_scale`) and running them on several workers
(`distribute`).
"""

import itertools


def setup_network(*, agents,
                  evals,
                  ps,
                  replay,
                  learner,
                  tensorplex,
                  loggerplex,
                  tensorboard,
                  session_config=None):
    """
        Sets up the communication between surreal
        components using symphony

        Args:
            agents, evals (list): list of symphony processes
            ps, replay, learner, tensorplex, loggerplex, tensorboard:
                symphony processes
    """
    if session_config is not None:
        ps.binds({'ps-frontend': session_config.ps.parameter_serving_frontend_port})
        ps.binds({'ps-backend': session_config.ps.parameter_serving_backend_port})

        replay.binds({'collector-frontend': session_config.replay.collector_frontend_port})
        replay.binds({'sampler-frontend': session_config.replay.sampler_frontend_port})
        replay.binds({'collector-backend': session_config.replay.collector_backend_port})
        replay.binds({'sampler-backend': session_config.replay.sampler_backend_port})

        learner.binds({'parameter-publish': session_config.ps.publish_port})
        learner.binds({'prefetch-queue': session_config.learner.prefetch_port})

        tensorplex.binds({'tensorplex': session_config.tensorplex.port})
        loggerplex.binds({'loggerplex': session_config.loggerplex.port})

        tensorboard.exposes({'tensorboard': session_config.tensorplex.tensorboard_port})

    else:
        ps.binds('ps-frontend')
        ps.binds('ps-backend')

        replay.binds('collector-frontend')
        replay.binds('sampler-frontend')
        replay.binds('collector-backend')
        replay.binds('sampler-backend')

        learner.binds('parameter-publish')
        learner.binds('prefetch-queue')

        tensorplex.binds('tensorplex')
        loggerplex.binds('loggerplex')

        tensorboard.exposes({'tensorboard': 6006})

    for proc in itertools.chain(agents, evals):
        proc.connects('ps-frontend')
        proc.connects('collector-frontend')

    ps.connects('parameter-publish')
    learner.connects('sampler-frontend')
    for proc in itertools.chain(agents, evals, [ps, replay, learner]):
        proc.connects('tensorplex')
        proc.connects('loggerplex')


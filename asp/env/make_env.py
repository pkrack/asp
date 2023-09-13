"""Helper functions to create alice and bob variants of an environment"""
from asp.env.wrappers import (
    AliceWrapper,
    AliceWrapperParameters,
    BobWrapper,
    BobWrapperParameters,
)


def make_alice_env(make_env, params: AliceWrapperParameters, rank, seed=0):
    """Transform a make_env function to a function that returns an alice
    environment."""

    def _init():
        init = make_env()
        env = init()
        return AliceWrapper(env, params)

    return _init


def make_bob_env(make_env, params: BobWrapperParameters, rank, seed=0):
    """Transform a make_env function to a function that returns a bob environment."""

    def _init():
        init = make_env()
        env = init()
        return BobWrapper(env, params)

    return _init

"""Define o ambiente :class:`CarEnv` e classes relacionadas."""


def init() -> None:
    """Realiza toda a inicialização necessária para o :class:`CarEnv`."""

    from . import actors
    actors.init()

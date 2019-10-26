import yaml
import logging
import logging.config
from pathlib import Path

from .saving import log_path


def setup_logging(run_config, log_config='logging.yml', default_level=logging.INFO) -> None:
    """
    Setup ``logging.config``

    Parameters
    ----------
    run_config : str
        Path to configuration file for run
    log_config : str
        Path to configuration file for logging
    default_level : int
        Default logging verbosity.
    """
    log_config = Path(log_config)

    if not log_config.exists():
        logging.basicConfig(level=default_level)
        logger = logging.getLogger('setup')
        logger.warning(f'"{log_config}" not found. Using basicConfig.')
        return

    with open(log_config, 'rt') as f:
        config = yaml.safe_load(f.read())

    # modify logging paths based on run config
    run_path = log_path(run_config)
    for _, handler in config['handlers'].items():
        if 'filename' in handler:
            handler['filename'] = str(run_path / handler['filename'])

    logging.config.dictConfig(config)


logging_level_dict = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}


def setup_logger(cls, verbose=0):
    logger = logging.getLogger(cls.__class__.__name__)
    if verbose not in logging_level_dict:
        raise KeyError(f'verbose option {verbose} for {cls} not valid. '
                       f'Valid options are {logging_level_dict.keys()}.')
    logger.setLevel(logging_level_dict[verbose])
    return logger

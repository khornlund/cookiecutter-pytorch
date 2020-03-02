import click
import yaml

from {{ cookiecutter.package_name }} import main
from {{ cookiecutter.package_name }}.utils import setup_logging


@click.group()
def cli():
    """
    CLI for {{ cookiecutter.package_name }}
    """
    pass


@cli.command()
@click.option(
    '-c',
    '--config-filename',
    default=['experiments/config.yml'],
    multiple=True,
    help=(
        'Path to training configuration file. If multiple are provided, runs will be '
        'executed in order'
    )
)
@click.option('-r', '--resume', default=None, type=str, help='path to checkpoint')
def train(config_filename, resume):
    """
    Entry point to start training run(s).
    """
    configs = [load_config(f) for f in config_filename]
    for config in configs:
        setup_logging(config)
        main.train(config, resume)


def load_config(filename: str) -> dict:
    """
    Load a configuration file as YAML.
    """
    with open(filename) as fh:
        config = yaml.safe_load(fh)
    return config

import logging
from dataclasses import dataclass
from multiprocessing import Queue
from logging.handlers import QueueHandler
from os.path import splitext, basename, join, dirname

from utils.i18n import init_gettext
from expressions.base import getExpressionLoader


# Context data class
@dataclass
class WorkerContext:
    log_queue: Queue = Queue()
    logger_app_name: str = splitext(basename(__file__))[0]
    logger_exp_name: str = getExpressionLoader(None).__name__
    formatter_app: logging.Formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%H:%M:%S"
)
    formatter_exp: logging.Formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(expression)s]: %(message)s", datefmt="%H:%M:%S"
)
    lang: str = "en"
    locale_dir: str = join(dirname(__file__), 'locales')
    domain: str = "app"


def setup_worker_context(worker_context: WorkerContext):
    init_gettext(worker_context.lang, worker_context.locale_dir, worker_context.domain)
    for name, formatter in [
        (worker_context.logger_app_name, worker_context.formatter_app),
        (worker_context.logger_exp_name, worker_context.formatter_exp),
    ]:
        handler = QueueHandler(worker_context.log_queue)
        handler.setFormatter(formatter)
        logging.getLogger(name).addHandler(handler)

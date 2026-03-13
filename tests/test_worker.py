"""Tests for utils/worker.py."""

import logging
import multiprocessing.queues
from dataclasses import fields
from multiprocessing import Queue
from logging.handlers import QueueHandler
from unittest.mock import patch

from utils.worker import WorkerContext, setup_worker_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(**overrides) -> WorkerContext:
    """Return a WorkerContext with sensible test defaults, allowing overrides."""
    ctx = WorkerContext(
        log_queue=Queue(),
        logger_app_name="test_app",
        logger_exp_name="test_exp",
        formatter_app=logging.Formatter("%(message)s"),
        formatter_exp=logging.Formatter("%(message)s"),
        lang="en",
        locale_dir="/fake/locales",
        domain="app",
    )
    for k, v in overrides.items():
        object.__setattr__(ctx, k, v)
    return ctx


def _get_queue_handlers(logger: logging.Logger) -> list[QueueHandler]:
    return [h for h in logger.handlers if isinstance(h, QueueHandler)]


# ---------------------------------------------------------------------------
# WorkerContext — dataclass structure
# ---------------------------------------------------------------------------

class TestWorkerContextDataclass:

    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        assert is_dataclass(WorkerContext)

    def test_has_log_queue_field(self):
        names = {f.name for f in fields(WorkerContext)}
        assert "log_queue" in names

    def test_has_logger_app_name_field(self):
        names = {f.name for f in fields(WorkerContext)}
        assert "logger_app_name" in names

    def test_has_logger_exp_name_field(self):
        names = {f.name for f in fields(WorkerContext)}
        assert "logger_exp_name" in names

    def test_has_formatter_app_field(self):
        names = {f.name for f in fields(WorkerContext)}
        assert "formatter_app" in names

    def test_has_formatter_exp_field(self):
        names = {f.name for f in fields(WorkerContext)}
        assert "formatter_exp" in names

    def test_has_lang_field(self):
        names = {f.name for f in fields(WorkerContext)}
        assert "lang" in names

    def test_has_locale_dir_field(self):
        names = {f.name for f in fields(WorkerContext)}
        assert "locale_dir" in names

    def test_has_domain_field(self):
        names = {f.name for f in fields(WorkerContext)}
        assert "domain" in names

    def test_log_queue_default_is_queue(self):
        ctx = WorkerContext()
        assert isinstance(ctx.log_queue, multiprocessing.queues.Queue)

    def test_formatter_app_default_is_formatter(self):
        ctx = WorkerContext()
        assert isinstance(ctx.formatter_app, logging.Formatter)

    def test_formatter_exp_default_is_formatter(self):
        ctx = WorkerContext()
        assert isinstance(ctx.formatter_exp, logging.Formatter)

    def test_lang_default_is_en(self):
        ctx = WorkerContext()
        assert ctx.lang == "en"

    def test_domain_default_is_app(self):
        ctx = WorkerContext()
        assert ctx.domain == "app"

    def test_logger_app_name_default_is_string(self):
        ctx = WorkerContext()
        assert isinstance(ctx.logger_app_name, str)

    def test_logger_exp_name_default_is_string(self):
        ctx = WorkerContext()
        assert isinstance(ctx.logger_exp_name, str)

    def test_locale_dir_default_is_string(self):
        ctx = WorkerContext()
        assert isinstance(ctx.locale_dir, str)

    def test_locale_dir_default_ends_with_locales(self):
        ctx = WorkerContext()
        assert ctx.locale_dir.endswith("locales")

    def test_formatter_app_default_format_contains_levelname(self):
        ctx = WorkerContext()
        assert "%(levelname)s" in ctx.formatter_app._fmt

    def test_formatter_exp_default_format_contains_expression(self):
        ctx = WorkerContext()
        assert "%(expression)s" in ctx.formatter_exp._fmt

    def test_custom_values_accepted(self):
        q = Queue()
        ctx = WorkerContext(log_queue=q, lang="ja", domain="myapp")
        assert ctx.log_queue is q
        assert ctx.lang == "ja"
        assert ctx.domain == "myapp"

    def test_two_default_instances_share_the_same_queue(self):
        """WorkerContext uses a bare Queue() as the field default (evaluated once
        at class-definition time), so all default instances share the same object.
        This test documents that behaviour; callers that need isolation must pass
        their own Queue explicitly."""
        ctx1 = WorkerContext()
        ctx2 = WorkerContext()
        assert ctx1.log_queue is ctx2.log_queue


# ---------------------------------------------------------------------------
# setup_worker_context
# ---------------------------------------------------------------------------

class TestSetupWorkerContext:

    def setup_method(self):
        """Remove any handlers added during the test to avoid cross-test pollution."""
        self._loggers_to_clean: list[str] = []

    def teardown_method(self):
        for name in self._loggers_to_clean:
            logger = logging.getLogger(name)
            logger.handlers = [h for h in logger.handlers
                                if not isinstance(h, QueueHandler)]

    def _run(self, ctx: WorkerContext | None = None):
        if ctx is None:
            ctx = _make_context()
        self._loggers_to_clean += [ctx.logger_app_name, ctx.logger_exp_name]
        with patch("utils.worker.init_gettext") as mock_i18n:
            setup_worker_context(ctx)
        return ctx, mock_i18n

    # --- init_gettext ---

    def test_init_gettext_called_once(self):
        ctx, mock_i18n = self._run()
        mock_i18n.assert_called_once()

    def test_init_gettext_receives_lang(self):
        ctx = _make_context(lang="fr")
        _, mock_i18n = self._run(ctx)
        args = mock_i18n.call_args
        assert "fr" in args.args or args.kwargs.get("lang") == "fr"

    def test_init_gettext_receives_locale_dir(self):
        ctx = _make_context(locale_dir="/my/locales")
        _, mock_i18n = self._run(ctx)
        args = mock_i18n.call_args
        assert "/my/locales" in args.args or args.kwargs.get("locale_dir") == "/my/locales"

    def test_init_gettext_receives_domain(self):
        ctx = _make_context(domain="myapp")
        _, mock_i18n = self._run(ctx)
        args = mock_i18n.call_args
        assert "myapp" in args.args or args.kwargs.get("domain") == "myapp"

    # --- handler registration: app logger ---

    def test_app_logger_gets_queue_handler(self):
        ctx, _ = self._run()
        handlers = _get_queue_handlers(logging.getLogger(ctx.logger_app_name))
        assert len(handlers) >= 1

    def test_app_logger_handler_uses_correct_queue(self):
        ctx, _ = self._run()
        handler = _get_queue_handlers(logging.getLogger(ctx.logger_app_name))[0]
        assert handler.queue is ctx.log_queue

    def test_app_logger_handler_has_app_formatter(self):
        ctx, _ = self._run()
        handler = _get_queue_handlers(logging.getLogger(ctx.logger_app_name))[0]
        assert handler.formatter is ctx.formatter_app

    # --- handler registration: exp logger ---

    def test_exp_logger_gets_queue_handler(self):
        ctx, _ = self._run()
        handlers = _get_queue_handlers(logging.getLogger(ctx.logger_exp_name))
        assert len(handlers) >= 1

    def test_exp_logger_handler_uses_correct_queue(self):
        ctx, _ = self._run()
        handler = _get_queue_handlers(logging.getLogger(ctx.logger_exp_name))[0]
        assert handler.queue is ctx.log_queue

    def test_exp_logger_handler_has_exp_formatter(self):
        ctx, _ = self._run()
        handler = _get_queue_handlers(logging.getLogger(ctx.logger_exp_name))[0]
        assert handler.formatter is ctx.formatter_exp

    # --- both loggers get separate handlers ---

    def test_app_and_exp_handlers_are_distinct_objects(self):
        ctx, _ = self._run()
        app_h = _get_queue_handlers(logging.getLogger(ctx.logger_app_name))[0]
        exp_h = _get_queue_handlers(logging.getLogger(ctx.logger_exp_name))[0]
        assert app_h is not exp_h

    def test_app_and_exp_formatters_differ(self):
        """Default formatters use different format strings; they must not be the same object."""
        ctx = _make_context(
            formatter_app=logging.Formatter("%(name)s %(message)s"),
            formatter_exp=logging.Formatter("%(expression)s %(message)s"),
        )
        self._run(ctx)
        app_h = _get_queue_handlers(logging.getLogger(ctx.logger_app_name))[0]
        exp_h = _get_queue_handlers(logging.getLogger(ctx.logger_exp_name))[0]
        assert app_h.formatter is not exp_h.formatter

    # --- idempotency / repeated calls ---

    def test_calling_twice_adds_two_handlers(self):
        """Each call appends a new handler; callers are responsible for avoiding duplicates."""
        ctx = _make_context()
        self._loggers_to_clean += [ctx.logger_app_name, ctx.logger_exp_name]
        with patch("utils.worker.init_gettext"):
            setup_worker_context(ctx)
            setup_worker_context(ctx)
        handlers = _get_queue_handlers(logging.getLogger(ctx.logger_app_name))
        assert len(handlers) == 2

    # --- different logger names do not cross-contaminate ---

    def test_different_exp_name_registers_separate_logger(self):
        ctx_a = _make_context(logger_app_name="app_a", logger_exp_name="exp_a")
        ctx_b = _make_context(logger_app_name="app_b", logger_exp_name="exp_b")
        self._loggers_to_clean += ["app_a", "exp_a", "app_b", "exp_b"]
        with patch("utils.worker.init_gettext"):
            setup_worker_context(ctx_a)
            setup_worker_context(ctx_b)
        assert len(_get_queue_handlers(logging.getLogger("exp_a"))) == 1
        assert len(_get_queue_handlers(logging.getLogger("exp_b"))) == 1

    def test_app_logger_name_not_registered_to_exp_logger(self):
        ctx = _make_context()
        self._run(ctx)
        exp_handlers = _get_queue_handlers(logging.getLogger(ctx.logger_exp_name))
        # None of the exp logger's handlers should carry the app formatter
        for h in exp_handlers:
            assert h.formatter is not ctx.formatter_app

"""
config/logging.py

Configures structlog for the project.
- JSON output in production (LOG_LEVEL != DEBUG)
- Colored, human-readable console output in development (LOG_LEVEL=DEBUG)

Call configure_logging() once at application startup (done in run_agent.py).

Usage:
    import structlog
    log = structlog.get_logger(__name__)
    log.info("node.enter", node="planner", trace_id=state["trace_id"])
"""

import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Set up structlog with appropriate renderer for the environment."""

    level = getattr(logging, log_level.upper(), logging.INFO)
    is_debug = level == logging.DEBUG

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_debug:
        # Human-readable colored output for local development
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Machine-readable JSON for production / CI
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

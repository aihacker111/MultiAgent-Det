from __future__ import annotations
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
from utils.logger import get_logger

logger = get_logger("event_bus")


class EventType(Enum):
    EPOCH_END = auto()
    MONITOR_ALERT = auto()
    MONITOR_PLATEAU = auto()
    MONITOR_EARLY_STOP = auto()
    VALIDATION_END = auto()
    ANALYSIS_COMPLETE = auto()
    PLAN_READY = auto()
    TRAINING_START = auto()
    TRAINING_COMPLETE = auto()
    TRAINING_FAILED = auto()
    EXPERIMENT_DUPLICATE = auto()
    SYSTEM_ERROR = auto()


@dataclass
class Event:
    type: EventType
    source: str
    data: Any = None
    metadata: dict = field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._history: list[Event] = []

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        with self._lock:
            handlers = self._subscribers[event_type]
            if handler in handlers:
                handlers.remove(handler)

    def publish(self, event: Event) -> None:
        with self._lock:
            handlers = list(self._subscribers[event.type])
            self._history.append(event)

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.error(f"Handler {handler.__name__} failed for {event.type}: {exc}")

    def get_history(self, event_type: EventType | None = None) -> list[Event]:
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if e.type == event_type]

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

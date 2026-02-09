# Python Conventions & Patterns Guide

> A practical quick-reference for starting and maintaining Python projects.
> Last updated: February 2026 · Targets Python 3.12+

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Naming Conventions](#2-naming-conventions)
3. [The Modern Toolchain](#3-the-modern-toolchain)
4. [Code Style & Formatting](#4-code-style--formatting)
5. [Type Hints](#5-type-hints)
6. [Design Patterns](#6-design-patterns)
7. [Pythonic Idioms](#7-pythonic-idioms)
8. [Data Modeling](#8-data-modeling)
9. [Error Handling](#9-error-handling)
10. [Logging](#10-logging)
11. [Testing](#11-testing)
12. [Async Patterns](#12-async-patterns)
13. [Configuration & Secrets](#13-configuration--secrets)
14. [Pre-commit & CI/CD](#14-pre-commit--cicd)
15. [Security Checklist](#15-security-checklist)
16. [Anti-Patterns to Avoid](#16-anti-patterns-to-avoid)

---

## 1. Project Structure

### Small project / script

```
my-tool/
├── pyproject.toml
├── README.md
├── src/
│   └── my_tool/
│       ├── __init__.py
│       ├── main.py
│       └── utils.py
└── tests/
    ├── __init__.py
    └── test_main.py
```

### Web backend (FastAPI / Flask / Django)

```
my-api/
├── pyproject.toml
├── README.md
├── alembic/                  # DB migrations
│   └── versions/
├── src/
│   └── my_api/
│       ├── __init__.py
│       ├── main.py           # app entrypoint
│       ├── config.py          # settings (pydantic-settings)
│       ├── dependencies.py    # DI helpers
│       ├── routers/           # route handlers
│       │   ├── __init__.py
│       │   ├── users.py
│       │   └── items.py
│       ├── services/          # business logic
│       │   ├── __init__.py
│       │   └── user_service.py
│       ├── repositories/      # DB access
│       │   ├── __init__.py
│       │   └── user_repo.py
│       ├── models/            # ORM models
│       │   ├── __init__.py
│       │   └── user.py
│       └── schemas/           # Pydantic request/response DTOs
│           ├── __init__.py
│           └── user.py
└── tests/
    ├── conftest.py
    ├── test_routers/
    └── test_services/
```

**Layer rule:** `routers → services → repositories → models`. Never skip layers.

### Library / package

```
my-lib/
├── pyproject.toml
├── README.md
├── LICENSE
├── docs/
├── src/
│   └── my_lib/
│       ├── __init__.py        # public API + __all__
│       ├── core.py
│       ├── _internals.py      # underscore = private module
│       └── py.typed           # PEP 561 marker for type checkers
└── tests/
```

### CLI application

```
my-cli/
├── pyproject.toml             # [project.scripts] section
├── README.md
├── src/
│   └── my_cli/
│       ├── __init__.py
│       ├── __main__.py        # python -m my_cli
│       ├── cli.py             # click/typer commands
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── init.py
│       │   └── run.py
│       └── core/
│           └── ...
└── tests/
```

### Data / ML project

```
my-ml-project/
├── pyproject.toml
├── README.md
├── data/
│   ├── raw/                   # immutable original data
│   ├── processed/             # cleaned data
│   └── .gitkeep
├── notebooks/                 # exploration only — not production code
├── src/
│   └── my_ml/
│       ├── __init__.py
│       ├── data/              # loading & preprocessing
│       ├── features/          # feature engineering
│       ├── models/            # model definitions
│       ├── training/          # training loops
│       └── evaluation/        # metrics & reporting
├── configs/                   # YAML/TOML experiment configs
├── models/                    # saved model artifacts (.gitignore these)
└── tests/
```

### Monorepo (uv workspaces)

```
my-org/
├── pyproject.toml             # workspace root
├── uv.lock                    # single lockfile for all packages
├── packages/
│   ├── core/
│   │   ├── pyproject.toml
│   │   └── src/core/...
│   ├── api/
│   │   ├── pyproject.toml     # depends on core
│   │   └── src/api/...
│   └── worker/
│       ├── pyproject.toml     # depends on core
│       └── src/worker/...
└── tests/
```

Root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["packages/*"]
```

### Key rules for all structures

- Always use `src/` layout — prevents accidental imports from project root.
- One top-level package per project.
- Tests mirror the source structure.
- Never commit: `.venv/`, `__pycache__/`, `.env`, `*.pyc`, `dist/`, model artifacts.

---

## 2. Naming Conventions

```python
# Files & modules
my_module.py                   # snake_case

# Classes
class UserService:             # PascalCase
class HTTPClient:              # Acronyms stay uppercase

# Functions & methods
def get_user_by_id():          # snake_case
def _internal_helper():        # single underscore = internal/private

# Variables
user_count = 42                # snake_case
_cache = {}                    # single underscore = internal

# Constants
MAX_RETRIES = 3                # UPPER_SNAKE_CASE
DEFAULT_TIMEOUT = 30

# Type variables (Python 3.12+)
type NumberLike = int | float  # PascalCase for type aliases

# Dunder (magic methods)
def __repr__(self):            # reserved for Python protocols
def __init__(self):

# Boolean variables & functions
is_active = True               # is_ / has_ / can_ / should_
has_permission = False
def is_valid():
```

### What NOT to do

```python
# ❌ Avoid
myVar = 1                      # camelCase for variables
class user_service:            # snake_case for classes
def GetUser():                 # PascalCase for functions
__my_var = "x"                 # double underscore (name mangling) — rarely needed
l = [1, 2, 3]                  # ambiguous single-letter names
```

---

## 3. The Modern Toolchain

### Recommended default stack (2025–2026)

| Purpose              | Tool              | Replaces                        |
| -------------------- | ----------------- | ------------------------------- |
| Package manager      | **uv**            | pip, pip-tools, pipx, pyenv     |
| Linter + formatter   | **ruff**          | flake8, black, isort, pyupgrade |
| Type checker (CI)    | **mypy / pyright** | —                              |
| Type checker (IDE)   | **pyright/Pylance** | —                             |
| Testing              | **pytest**        | unittest                        |
| Task runner          | **just** or **make** | scripts, Makefile            |

### Minimal `pyproject.toml`

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "ruff",
    "mypy",
    "pre-commit",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# ── Ruff ────────────────────────────────────────
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = [
    "E", "W",     # pycodestyle
    "F",           # pyflakes
    "I",           # isort
    "UP",          # pyupgrade
    "N",           # pep8-naming
    "S",           # bandit (security)
    "B",           # bugbear
    "A",           # builtins shadowing
    "SIM",         # simplify
    "ANN",         # annotations
    "RUF",         # ruff-specific
]
ignore = ["ANN101", "ANN102"]  # self/cls type hints

[tool.ruff.lint.isort]
known-first-party = ["my_project"]

# ── Mypy ────────────────────────────────────────
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true

# ── Pytest ──────────────────────────────────────
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --strict-markers"

[tool.coverage.run]
branch = true
source = ["src"]

[tool.coverage.report]
fail_under = 80
```

### Daily commands

```bash
uv init my-project             # scaffold a new project
uv add fastapi pydantic        # add dependencies
uv add --dev pytest ruff mypy  # add dev dependencies
uv sync                        # install everything from lockfile
uv run pytest                  # run in project's venv
uv run ruff check .            # lint
uv run ruff format .           # format
uv run mypy src/               # type check
```

---

## 4. Code Style & Formatting

### Imports — always in this order

```python
# 1. Standard library
import os
from pathlib import Path
from typing import Any

# 2. Third-party
from fastapi import FastAPI
from pydantic import BaseModel

# 3. Local / first-party
from my_project.services import UserService
from my_project.models import User
```

Ruff handles sorting automatically (`I` rule).

### Line length

88 characters (ruff/black default). Use implicit line continuation:

```python
# ✅ Good — implicit continuation inside parentheses
result = (
    some_long_function_name(
        argument_one,
        argument_two,
        argument_three,
    )
)

# ❌ Avoid — backslash continuation
result = some_long_function_name(argument_one, \
    argument_two)
```

### Trailing commas

Always use trailing commas in multi-line structures:

```python
# ✅ Cleaner diffs, easier to extend
my_list = [
    "first",
    "second",
    "third",      # ← trailing comma
]

def create_user(
    name: str,
    email: str,
    role: str = "user",   # ← trailing comma
) -> User:
```

### Docstrings (Google style)

```python
def fetch_user(user_id: UUID, *, include_deleted: bool = False) -> User | None:
    """Fetch a user by their unique ID.

    Looks up the user in the primary database. Returns None if no
    matching user is found.

    Args:
        user_id: The unique identifier of the user.
        include_deleted: If True, also returns soft-deleted users.

    Returns:
        The matching User object, or None if not found.

    Raises:
        DatabaseConnectionError: If the database is unreachable.
    """
```

**Rule:** Don't repeat type information in docstrings — that's what type hints are for. Describe *what* and *why*, not *what type*.

---

## 5. Type Hints

### Basics

```python
# Variables (usually inferred — annotate when unclear)
name: str = "Alice"
scores: list[int] = [90, 85, 92]
config: dict[str, Any] = {}

# Functions — always annotate parameters and return
def greet(name: str, excited: bool = False) -> str:
    return f"Hello, {name}{'!' if excited else '.'}"

# None returns
def log_event(event: str) -> None:
    print(event)

# Optional / nullable
def find_user(user_id: int) -> User | None:   # Python 3.10+ union syntax
    ...
```

### Modern syntax (Python 3.12+)

```python
# ✅ New style — PEP 695
type Vector = list[float]
type Result[T] = T | None
type Callback[T] = Callable[[T], None]

def first[T](items: list[T]) -> T | None:
    return items[0] if items else None

class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

# ❌ Old style — still works but verbose
from typing import TypeVar, Generic
T = TypeVar("T")
class Stack(Generic[T]):
    ...
```

### Common patterns

```python
from typing import Any, Protocol
from collections.abc import Callable, Iterator, Sequence

# Callable
Handler = Callable[[Request], Response]

# Protocol (structural typing — preferred over ABC)
class Readable(Protocol):
    def read(self, n: int = -1) -> bytes: ...

def process(source: Readable) -> None:
    data = source.read()        # works with any object that has .read()

# TypedDict — for unstructured dict data
from typing import TypedDict

class UserDict(TypedDict):
    name: str
    email: str
    age: int | None
```

---

## 6. Design Patterns

### Decorator

The most common pattern in Python. Use for cross-cutting concerns.

```python
import functools
import time
from typing import Any
from collections.abc import Callable

# Basic decorator with wraps
def timer[T](func: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def process_data(data: list[int]) -> int:
    return sum(data)

# Decorator with parameters
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator[T](func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator

@retry(max_attempts=5, delay=2.0)
def call_external_api() -> dict:
    ...
```

### Factory

Prefer functions over factory classes.

```python
# Simple factory function
def create_storage(kind: str, **kwargs) -> Storage:
    match kind:
        case "s3":
            return S3Storage(**kwargs)
        case "local":
            return LocalStorage(**kwargs)
        case "gcs":
            return GCSStorage(**kwargs)
        case _:
            raise ValueError(f"Unknown storage type: {kind}")

# Factory via classmethod (alternative constructors)
class User:
    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "User":
        return cls(name=data["name"], email=data["email"])

    @classmethod
    def from_db_row(cls, row: tuple) -> "User":
        return cls(name=row[0], email=row[1])
```

### Strategy

Use callables — no need for a class hierarchy.

```python
from collections.abc import Callable
from dataclasses import dataclass

# Strategy as functions
def price_fixed(base: float, quantity: int) -> float:
    return base * quantity

def price_bulk(base: float, quantity: int) -> float:
    discount = 0.1 if quantity > 100 else 0
    return base * quantity * (1 - discount)

PricingStrategy = Callable[[float, int], float]

def calculate_total(
    base: float,
    quantity: int,
    strategy: PricingStrategy = price_fixed,
) -> float:
    return strategy(base, quantity)

# Usage
total = calculate_total(10.0, 150, strategy=price_bulk)
```

### Repository

Decouple business logic from data access.

```python
from abc import ABC, abstractmethod
# Or use Protocol for structural typing:
# from typing import Protocol

class UserRepository(ABC):
    @abstractmethod
    def get_by_id(self, user_id: UUID) -> User | None: ...

    @abstractmethod
    def save(self, user: User) -> None: ...

    @abstractmethod
    def delete(self, user_id: UUID) -> None: ...


class SQLUserRepository(UserRepository):
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_id(self, user_id: UUID) -> User | None:
        return self._session.get(UserModel, user_id)

    def save(self, user: User) -> None:
        self._session.add(user)
        self._session.commit()

    def delete(self, user_id: UUID) -> None:
        ...


# In tests, swap with a fake:
class FakeUserRepository(UserRepository):
    def __init__(self) -> None:
        self._users: dict[UUID, User] = {}

    def get_by_id(self, user_id: UUID) -> User | None:
        return self._users.get(user_id)
    ...
```

### Dependency Injection (lightweight)

Python doesn't need a DI framework — just pass dependencies as arguments.

```python
# Constructor injection
class OrderService:
    def __init__(
        self,
        user_repo: UserRepository,
        payment_gateway: PaymentGateway,
        notifier: Notifier,
    ) -> None:
        self._user_repo = user_repo
        self._payment = payment_gateway
        self._notifier = notifier

# FastAPI style — function-level injection
from fastapi import Depends

def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_repo(db: Session = Depends(get_db)) -> UserRepository:
    return SQLUserRepository(db)

@router.get("/users/{user_id}")
def get_user(
    user_id: UUID,
    repo: UserRepository = Depends(get_user_repo),
) -> User:
    return repo.get_by_id(user_id)
```

### Singleton

Don't create Singleton classes — use module-level instances.

```python
# ✅ Python's module system IS a singleton
# config.py
from my_app.settings import Settings

settings = Settings()    # created once on first import

# anywhere else:
from my_app.config import settings    # same instance every time
```

### Context Manager

Use for any resource that needs cleanup.

```python
from contextlib import contextmanager

# Class-based
class DatabaseConnection:
    def __enter__(self) -> "DatabaseConnection":
        self._conn = create_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._conn.close()

# Function-based (simpler, preferred)
@contextmanager
def timed_operation(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.4f}s")

with timed_operation("data processing"):
    process_data()

# Async version
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_client():
    client = httpx.AsyncClient()
    try:
        yield client
    finally:
        await client.aclose()
```

### Observer / Event system

```python
from collections.abc import Callable
from dataclasses import dataclass, field

@dataclass
class EventBus:
    _listeners: dict[str, list[Callable]] = field(default_factory=dict)

    def on(self, event: str, callback: Callable) -> None:
        self._listeners.setdefault(event, []).append(callback)

    def emit(self, event: str, **data) -> None:
        for callback in self._listeners.get(event, []):
            callback(**data)

# Usage
bus = EventBus()
bus.on("user_created", lambda **d: send_welcome_email(d["email"]))
bus.on("user_created", lambda **d: log_event("signup", d))
bus.emit("user_created", email="alice@example.com", name="Alice")
```

### Adapter

Normalize third-party APIs behind a consistent interface.

```python
class PaymentProvider(Protocol):
    def charge(self, amount_cents: int, currency: str) -> str: ...

class StripeAdapter:
    def __init__(self, client: stripe.Client) -> None:
        self._client = client

    def charge(self, amount_cents: int, currency: str) -> str:
        intent = self._client.payment_intents.create(
            amount=amount_cents, currency=currency,
        )
        return intent.id

class PayPalAdapter:
    def __init__(self, client: paypal.Client) -> None:
        self._client = client

    def charge(self, amount_cents: int, currency: str) -> str:
        order = self._client.create_order(amount_cents / 100, currency)
        return order["id"]
```

---

## 7. Pythonic Idioms

### EAFP over LBYL

```python
# ✅ EAFP — Easier to Ask Forgiveness than Permission
try:
    value = data["key"]
except KeyError:
    value = default

# ❌ LBYL — Look Before You Leap (less Pythonic)
if "key" in data:
    value = data["key"]
else:
    value = default

# Even simpler when applicable:
value = data.get("key", default)
```

### Comprehensions

```python
# List comprehension
active_users = [u for u in users if u.is_active]

# Dict comprehension
user_map = {u.id: u for u in users}

# Set comprehension
unique_emails = {u.email.lower() for u in users}

# Generator expression (lazy — use for large data)
total = sum(order.total for order in orders)

# ❌ Don't use comprehensions for side effects
# Bad: [print(x) for x in items]
# Good:
for x in items:
    print(x)
```

### Generators for lazy iteration

```python
def read_large_file(path: Path):
    """Yields lines without loading entire file into memory."""
    with open(path) as f:
        for line in f:
            yield line.strip()

def batched(iterable, n: int):
    """Yield successive n-sized chunks."""
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch

# Pipeline style
results = (
    process(item)
    for item in read_large_file(path)
    if item  # skip empty lines
)
```

### Unpacking & walrus operator

```python
# Tuple unpacking
first, *rest = [1, 2, 3, 4]           # first=1, rest=[2,3,4]
head, *_, tail = [1, 2, 3, 4, 5]      # head=1, tail=5

# Dictionary unpacking
defaults = {"timeout": 30, "retries": 3}
overrides = {"timeout": 60}
config = {**defaults, **overrides}     # timeout=60, retries=3

# Walrus operator (:=) — assign inside expressions
while chunk := file.read(8192):
    process(chunk)

if (match := pattern.search(text)) is not None:
    print(match.group())
```

### Duck typing

```python
# ✅ Don't check types — check capabilities
def save(obj):
    obj.save()    # works with anything that has .save()

# When you need explicit contracts, use Protocol:
class Saveable(Protocol):
    def save(self) -> None: ...

def save(obj: Saveable) -> None:
    obj.save()
```

### Composition over inheritance

```python
# ✅ Composition — flexible, testable
class NotificationService:
    def __init__(self, sender: EmailSender, template: TemplateEngine) -> None:
        self._sender = sender
        self._template = template

    def notify(self, user: User, event: str) -> None:
        body = self._template.render(event, user=user)
        self._sender.send(user.email, body)

# ❌ Deep inheritance — fragile, hard to change
class BaseNotifier:
    ...
class EmailNotifier(BaseNotifier):
    ...
class TemplatedEmailNotifier(EmailNotifier):
    ...
```

---

## 8. Data Modeling

### When to use what

| Need                        | Use                              |
| --------------------------- | -------------------------------- |
| Internal data / domain objects | `@dataclass`                   |
| API input/output / validation | `pydantic.BaseModel`           |
| Immutable config / DTOs     | `@dataclass(frozen=True)`        |
| Simple grouping of constants | `Enum`                          |
| Unstructured dicts with known keys | `TypedDict`               |
| Database rows               | ORM model (SQLAlchemy / etc.)    |

### Dataclass (internal objects)

```python
from dataclasses import dataclass, field
from uuid import UUID, uuid4

@dataclass(frozen=True, slots=True)  # immutable + memory efficient
class Money:
    amount: int
    currency: str = "USD"

@dataclass(slots=True, kw_only=True)  # keyword-only for 3+ fields
class Order:
    id: UUID = field(default_factory=uuid4)
    customer_id: UUID
    items: list[str] = field(default_factory=list)
    total: Money = field(default_factory=lambda: Money(0))

    def add_item(self, item: str) -> None:
        self.items.append(item)
```

### Pydantic (API boundaries)

```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class CreateUserRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str
    age: int = Field(ge=0, le=150)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email")
        return v.lower()

class UserResponse(BaseModel):
    id: UUID
    name: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}  # allows ORM → Pydantic
```

### Enums

```python
from enum import Enum, auto, StrEnum

class Status(StrEnum):       # string values — great for APIs
    ACTIVE = auto()          # "active"
    INACTIVE = auto()        # "inactive"
    SUSPENDED = auto()       # "suspended"

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

# Usage
user.status = Status.ACTIVE
if user.status == Status.SUSPENDED:
    ...
```

---

## 9. Error Handling

### Define domain exceptions

```python
# exceptions.py
class AppError(Exception):
    """Base for all application errors."""

class NotFoundError(AppError):
    """Raised when a requested resource doesn't exist."""

class ValidationError(AppError):
    """Raised when input data is invalid."""

class PermissionDeniedError(AppError):
    """Raised when the user lacks required permissions."""
```

### Rules

```python
# ✅ Catch specific exceptions
try:
    user = repo.get_by_id(user_id)
except NotFoundError:
    return {"error": "User not found"}, 404

# ✅ Catch at boundaries (routers, CLI entry points), not everywhere
# ✅ Let unexpected errors bubble up — don't swallow them

# ❌ Never do this
try:
    something()
except:           # bare except catches SystemExit, KeyboardInterrupt
    pass          # silently swallowing errors

# ❌ Avoid this too
try:
    something()
except Exception:
    pass          # slightly better but still hides bugs

# ✅ If you must catch broadly, log it
try:
    something()
except Exception:
    logger.exception("Unexpected error in something()")
    raise
```

---

## 10. Logging

### Setup with structlog (recommended for services)

```python
import structlog
import logging

def setup_logging(*, json_output: bool = True, level: str = "INFO") -> None:
    """Call once at application startup."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=level, format="%(message)s")
```

### Usage

```python
import structlog

logger = structlog.get_logger()

# Bound context — all subsequent logs include user_id
log = logger.bind(user_id=user.id)
log.info("order_created", order_id=order.id, total=order.total)
log.warning("payment_retry", attempt=3)

# Request-scoped context (FastAPI middleware)
structlog.contextvars.bind_contextvars(request_id=str(uuid4()))
# Now every log in this request includes request_id automatically
```

### For scripts / small projects: stdlib logging

```python
import logging

logger = logging.getLogger(__name__)    # one logger per module

logger.info("Processing %s items", len(items))  # ← lazy formatting

# ❌ Never use f-strings in log calls — they evaluate even if level is disabled
logger.debug(f"Big object: {expensive_repr()}")     # BAD
logger.debug("Big object: %s", expensive_repr())    # GOOD (lazy)
```

---

## 11. Testing

### Structure: Arrange – Act – Assert

```python
# tests/test_user_service.py
import pytest
from my_app.services import UserService

class TestUserService:
    def test_create_user_returns_user_with_id(self):
        # Arrange
        repo = FakeUserRepository()
        service = UserService(repo=repo)

        # Act
        user = service.create(name="Alice", email="alice@test.com")

        # Assert
        assert user.id is not None
        assert user.name == "Alice"

    def test_create_user_rejects_duplicate_email(self):
        repo = FakeUserRepository()
        service = UserService(repo=repo)
        service.create(name="Alice", email="alice@test.com")

        with pytest.raises(ValidationError, match="email already exists"):
            service.create(name="Bob", email="alice@test.com")
```

### Fixtures

```python
# tests/conftest.py — shared fixtures
import pytest

@pytest.fixture
def db_session():
    """Provides a clean database session per test."""
    session = create_test_session()
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def user_repo(db_session):
    return SQLUserRepository(db_session)

@pytest.fixture
def sample_user():
    return User(name="Test User", email="test@example.com")
```

### Parametrize

```python
@pytest.mark.parametrize(
    ("input_email", "expected_valid"),
    [
        ("user@example.com", True),
        ("user@sub.example.com", True),
        ("invalid", False),
        ("", False),
        ("@no-local.com", False),
    ],
)
def test_email_validation(input_email: str, expected_valid: bool):
    assert is_valid_email(input_email) == expected_valid
```

### Markers and coverage

```python
# Custom markers — define in pyproject.toml [tool.pytest.ini_options]
@pytest.mark.slow
def test_full_pipeline():
    ...

@pytest.mark.integration
def test_database_connection():
    ...

# Run subsets
# uv run pytest -m "not slow"
# uv run pytest --cov=src --cov-report=term-missing
```

---

## 12. Async Patterns

```python
import asyncio
import httpx

# ✅ Concurrent I/O with gather
async def fetch_all(urls: list[str]) -> list[str]:
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [r.text for r in responses]

# ✅ Async context manager for resources
async def lifespan(app: FastAPI):
    # Startup
    app.state.db = await create_pool()
    yield
    # Shutdown
    await app.state.db.close()

# ❌ Don't block the event loop
async def bad_handler():
    data = requests.get("https://...")      # BLOCKS everything
    result = heavy_computation()            # BLOCKS everything

# ✅ Offload blocking work
async def good_handler():
    data = await httpx_client.get("https://...")            # async I/O
    result = await asyncio.to_thread(heavy_computation)     # thread pool
```

### Rules

- Use `async`/`await` at the edges (routes, handlers). Keep core logic sync when possible.
- Never mix `requests` with `asyncio` — use `httpx.AsyncClient`.
- Use `asyncio.gather()` for concurrent tasks, not sequential awaits.
- Use `asyncio.TaskGroup()` (Python 3.11+) for structured concurrency.

---

## 13. Configuration & Secrets

### Settings with pydantic-settings

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "my-app"
    debug: bool = False
    database_url: str
    redis_url: str = "redis://localhost:6379"
    secret_key: str

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()   # reads from environment variables / .env file
```

### Environment files

```bash
# .env.example — commit this (no real values)
APP_NAME=my-app
DEBUG=false
DATABASE_URL=postgresql://user:pass@localhost/dbname
REDIS_URL=redis://localhost:6379
SECRET_KEY=change-me-in-production

# .env — NEVER commit this (add to .gitignore)
DATABASE_URL=postgresql://real-user:real-pass@db-host/prod
SECRET_KEY=super-secret-production-key
```

### Rules

- Config via environment variables (12-factor).
- `.env` for local development only — never commit it.
- Commit `.env.example` with safe placeholder values.
- Use cloud secret managers (AWS Secrets Manager, Vault) in production.
- Never hardcode secrets, API keys, or passwords in source code.

---

## 14. Pre-commit & CI/CD

### `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
      - id: detect-private-key

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.0
    hooks:
      - id: ruff            # lint (runs before format)
        args: [--fix]
      - id: ruff-format     # format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.14.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic]  # add stubs as needed
```

```bash
uv run pre-commit install       # setup (once per repo clone)
uv run pre-commit run --all-files  # manual full run
```

### GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --dev
      - run: uv run ruff check --output-format=github .
      - run: uv run ruff format --check .

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --dev
      - run: uv run mypy src/

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync --dev
      - run: uv run pytest --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v5
        if: matrix.python-version == '3.13'
```

---

## 15. Security Checklist

| Area                  | Do                                           | Don't                              |
| --------------------- | -------------------------------------------- | ---------------------------------- |
| **Secrets**           | Use env vars + secret manager                | Hardcode in source                 |
| **SQL**               | Use parameterized queries / ORM              | String-format SQL                  |
| **User input**        | Validate with Pydantic at boundaries         | Trust raw input                    |
| **Dependencies**      | Run `pip-audit`, enable Dependabot           | Ignore vulnerability alerts        |
| **Deserialization**   | Use JSON / Pydantic                          | Use `pickle` on untrusted data     |
| **Subprocess**        | Use `subprocess.run(["cmd", arg])`           | Use `shell=True` with user input   |
| **SAST**              | Run `ruff` S rules + `bandit` in CI          | Skip static analysis               |
| **Lockfile**          | Commit `uv.lock` with hashes                | Use unpinned `requirements.txt`    |
| **Publishing**        | Use PyPI Trusted Publishing (OIDC)           | Store PyPI tokens in env vars      |
| **Updates**           | Enable Dependabot or Renovate                | Let deps go stale for months       |

---

## 16. Anti-Patterns to Avoid

```python
# ❌ Mutable default argument
def add_item(item, items=[]):     # list is shared across calls!
    items.append(item)
    return items

# ✅ Fix
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

```python
# ❌ Bare except
try:
    do_something()
except:
    pass

# ✅ Catch specific exceptions
try:
    do_something()
except ValueError as e:
    logger.warning("Invalid value: %s", e)
```

```python
# ❌ God class — does everything
class UserManager:
    def create_user(self): ...
    def send_email(self): ...
    def generate_report(self): ...
    def sync_to_crm(self): ...

# ✅ Split by responsibility
class UserService: ...
class EmailService: ...
class ReportService: ...
```

```python
# ❌ Deep inheritance
class Animal: ...
class Mammal(Animal): ...
class Domestic(Mammal): ...
class Pet(Domestic): ...
class Dog(Pet): ...

# ✅ Composition + flat hierarchy
class Dog:
    def __init__(self, behavior: DomesticBehavior, diet: Diet): ...
```

```python
# ❌ Pydantic for everything (6.5x slower than dataclass)
class InternalPoint(BaseModel):   # overkill for internal data
    x: float
    y: float

# ✅ Dataclass for internal, Pydantic at boundaries
@dataclass(slots=True)
class InternalPoint:
    x: float
    y: float
```

```python
# ❌ Star imports
from os.path import *
from my_module import *

# ✅ Explicit imports
from pathlib import Path
from my_module import specific_function
```

```python
# ❌ Mixing sync and async
async def handler():
    result = requests.get(url)           # blocks the event loop!

# ✅ Use async-compatible libraries
async def handler():
    async with httpx.AsyncClient() as client:
        result = await client.get(url)
```

### More things to avoid

- Over-engineering: Don't add abstractions until you need them.
- Java-style getters/setters: Use `@property` only when you need computed values or validation.
- `type: ignore` without explanation: Always add a reason comment.
- Ignoring type checker errors: Fix them or document why they're false positives.
- Not using `__all__` in library `__init__.py`: It controls your public API.

---

> **Start simple, add complexity only when needed. Python rewards clarity over cleverness.**

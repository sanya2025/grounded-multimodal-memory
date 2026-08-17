# 5. Module-level Pydantic models in the FastAPI app

Status: Accepted

## Context

`grounded_memory/api/main.py` originally defined its request/response models
(`ObserveRequest`, `QueryRequest`, etc.) as classes local to `create_app()`,
alongside `from __future__ import annotations` at the top of the file. Under
postponed evaluation of annotations, a parameter annotation like
`req: ObserveRequest` becomes the *string* `"ObserveRequest"` at runtime, and
FastAPI resolves it via `typing.get_type_hints`, which looks the name up in
the function's `__globals__` — the module namespace. A class defined inside
`create_app()` never appears there, so resolution silently failed. FastAPI
then couldn't tell the parameter was a Pydantic model and treated it as a
plain (query) parameter instead of a request body. Every `POST /observe` and
`POST /query` call returned `422 Unprocessable Entity` — `{'loc': ['query',
'req'], 'msg': 'Field required'}` — because the JSON body was never read.

## Decision

Move `ObserveRequest`, `ObserveResponse`, `QueryRequest`, `Evidence`, and
`QueryResponse` to module scope in `main.py`, above `create_app()`. `pydantic`
is imported at module level too — safe, since `pydantic` is a base dependency
(unlike `fastapi`, which stays a lazy import inside `create_app()` so the
module can still be imported when the `api` extra isn't installed).

## Consequences

- Request bodies resolve correctly; the endpoints work as intended.
- Any future request/response model for this app must be defined at module
  scope, not nested inside `create_app()` or any other function — nesting
  reintroduces the same silent-failure mode under
  `from __future__ import annotations`.
- This is a general trap, not specific to this file: any FastAPI (or
  Pydantic-validated) app using postponed annotation evaluation must define
  its models somewhere `get_type_hints` can actually find them.

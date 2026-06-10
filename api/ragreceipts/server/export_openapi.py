"""Print the OpenAPI 3.1 schema without starting the server or building deps.

FastAPI emits OpenAPI 3.1.0 by default and serves it at /openapi.json; app.openapi()
returns the same document programmatically (verified:
https://fastapi.tiangolo.com/how-to/extending-openapi/). Used by web/ codegen:

    cd api && uv run python -m ragreceipts.server.export_openapi > ../web/openapi.json
"""

import json

from ragreceipts.server.app import create_app


def main() -> None:
    print(json.dumps(create_app().openapi(), indent=2))


if __name__ == "__main__":
    main()

import createClient from "openapi-fetch";
import type { paths } from "./schema";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// Typed client generated from FastAPI's OpenAPI 3.1 schema.
// Usage verified: https://openapi-ts.dev/openapi-fetch/
export const api = createClient<paths>({ baseUrl: API_BASE });

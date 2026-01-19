## Goal
Implement a user-initiated logout flow so authenticated users can clear their session and return to the login screen.

## Context
- Backend uses FastAPI `SessionMiddleware` with cookie-based sessions.
- Auth endpoints live in `backend/main.py`:
  - `POST /auth/register`
  - `POST /auth/login`
  - `POST /auth/logout` (should clear the session)
  - `GET /me`
- Frontend UI is in `frontend/index.html` with logic in `frontend/app.js`.

## Implementation plan
1) Backend (verify/ensure logout endpoint exists)
   - In `backend/main.py`, confirm `POST /auth/logout` exists.
   - If missing, add it near the other auth routes:
     - `req.session.clear()`
     - return `{ "ok": true }`

2) Frontend UI (add a logout control)
   - Add a logout button to `frontend/index.html` in a visible header area.
   - Give it a stable id (e.g. `logoutButton`) for wiring.
   - Optional: add a small status node for logout errors (e.g. `logoutStatus`).

3) Frontend logic (call logout API)
   - In `frontend/app.js`, attach a click handler:
     - `fetch("/auth/logout", { method: "POST", credentials: "same-origin" })`
     - On success: `location.replace("/auth.html")` to avoid back-navigation.
     - On failure: show a user-facing error (status node or `alert`).
   - Disable inputs/buttons while the logout request is running.

4) Styling
   - Update `frontend/room.css` to position the logout button in the header.
   - Keep layout responsive; reuse existing button styles when possible.

## Done when
- Clicking logout clears the session and redirects to `/auth.html`.
- Reloading `/` or hitting `/me` after logout returns 401 and routes to login.
- No regressions to login, chat, or PDF upload flows.

## Manual check
1) Register or login at `/auth.html`.
2) Visit `/`, click the logout button.
3) Confirm redirect to `/auth.html`.
4) Refresh `/` and verify it redirects to login (session cleared).

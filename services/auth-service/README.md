# Auth Service (Keycloak)

We use the official `quay.io/keycloak/keycloak` image with our exported realm.

## Local dev
`realm-export.json` is auto-imported on startup via docker-compose.

## Production
- Run Keycloak behind ALB (HTTPS), backed by RDS Postgres.
- Replace `distribute-backend` client secret using AWS Secrets Manager.
- Set `KC_HOSTNAME=auth.distrebute.com`, `KC_PROXY=edge`.

## Defaults shipped
- Realm: `distribute`
- Web client: `distribute-web` (public + PKCE)
- Backend client: `distribute-backend` (confidential, service account)
- Demo user: `demo / demo123` (DELETE before production)
- Roles: `user`, `creator`, `admin`

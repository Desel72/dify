"""Testcontainers integration tests for OAuthServerService."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch
from uuid import uuid4

import pytest
from werkzeug.exceptions import BadRequest

from extensions.ext_redis import redis_client
from models.model import OAuthProviderApp
from services.oauth_server import (
    OAUTH_ACCESS_TOKEN_REDIS_KEY,
    OAUTH_AUTHORIZATION_CODE_REDIS_KEY,
    OAUTH_REFRESH_TOKEN_REDIS_KEY,
    OAuthGrantType,
    OAuthServerService,
)


class TestOAuthServerServiceGetProviderApp:
    """DB-backed tests for get_oauth_provider_app."""

    def _create_oauth_provider_app(self, db_session_with_containers, *, client_id: str) -> OAuthProviderApp:
        app = OAuthProviderApp(
            app_icon="icon.png",
            client_id=client_id,
            client_secret=str(uuid4()),
            app_label={"en-US": "Test OAuth App"},
            redirect_uris=["https://example.com/callback"],
            scope="read",
        )
        db_session_with_containers.add(app)
        db_session_with_containers.commit()
        return app

    def test_get_oauth_provider_app_returns_app_when_exists(self, db_session_with_containers):
        client_id = f"client-{uuid4()}"
        created = self._create_oauth_provider_app(db_session_with_containers, client_id=client_id)

        result = OAuthServerService.get_oauth_provider_app(client_id)

        assert result is not None
        assert result.client_id == client_id
        assert result.id == created.id

    def test_get_oauth_provider_app_returns_none_when_not_exists(self, db_session_with_containers):
        result = OAuthServerService.get_oauth_provider_app(f"nonexistent-{uuid4()}")

        assert result is None


class TestOAuthServerServiceTokenOperations:
    """Real Redis-backed tests for token sign/validate operations."""

    def test_sign_authorization_code_stores_in_redis(self, flask_req_ctx_with_containers):
        client_id = f"client-{uuid4()}"
        user_id = f"user-{uuid4()}"

        code = OAuthServerService.sign_oauth_authorization_code(client_id, user_id)

        assert code is not None
        key = OAUTH_AUTHORIZATION_CODE_REDIS_KEY.format(client_id=client_id, code=code)
        stored = redis_client.get(key)
        assert stored is not None
        assert stored.decode() == user_id

        # Cleanup
        redis_client.delete(key)

    def test_sign_access_token_raises_bad_request_for_invalid_code(self, flask_req_ctx_with_containers):
        with pytest.raises(BadRequest, match="invalid code"):
            OAuthServerService.sign_oauth_access_token(
                grant_type=OAuthGrantType.AUTHORIZATION_CODE,
                code=f"bad-code-{uuid4()}",
                client_id=f"client-{uuid4()}",
            )

    def test_sign_access_token_issues_tokens_for_valid_code(self, flask_req_ctx_with_containers):
        client_id = f"client-{uuid4()}"
        user_id = f"user-{uuid4()}"

        # Store an authorization code first
        code = OAuthServerService.sign_oauth_authorization_code(client_id, user_id)

        # Exchange code for tokens
        access_token, refresh_token = OAuthServerService.sign_oauth_access_token(
            grant_type=OAuthGrantType.AUTHORIZATION_CODE,
            code=code,
            client_id=client_id,
        )

        assert access_token is not None
        assert refresh_token is not None

        # Verify authorization code was deleted
        code_key = OAUTH_AUTHORIZATION_CODE_REDIS_KEY.format(client_id=client_id, code=code)
        assert redis_client.get(code_key) is None

        # Verify access token was stored
        access_key = OAUTH_ACCESS_TOKEN_REDIS_KEY.format(client_id=client_id, token=access_token)
        assert redis_client.get(access_key) is not None

        # Verify refresh token was stored
        refresh_key = OAUTH_REFRESH_TOKEN_REDIS_KEY.format(client_id=client_id, token=refresh_token)
        assert redis_client.get(refresh_key) is not None

        # Cleanup
        redis_client.delete(access_key, refresh_key)

    def test_sign_access_token_raises_bad_request_for_invalid_refresh_token(self, flask_req_ctx_with_containers):
        with pytest.raises(BadRequest, match="invalid refresh token"):
            OAuthServerService.sign_oauth_access_token(
                grant_type=OAuthGrantType.REFRESH_TOKEN,
                refresh_token=f"stale-{uuid4()}",
                client_id=f"client-{uuid4()}",
            )

    def test_sign_access_token_issues_new_token_for_valid_refresh(self, flask_req_ctx_with_containers):
        client_id = f"client-{uuid4()}"
        user_id = f"user-{uuid4()}"

        # Create initial tokens
        code = OAuthServerService.sign_oauth_authorization_code(client_id, user_id)
        _, refresh_token = OAuthServerService.sign_oauth_access_token(
            grant_type=OAuthGrantType.AUTHORIZATION_CODE,
            code=code,
            client_id=client_id,
        )

        # Use refresh token to get new access token
        new_access_token, returned_refresh = OAuthServerService.sign_oauth_access_token(
            grant_type=OAuthGrantType.REFRESH_TOKEN,
            refresh_token=refresh_token,
            client_id=client_id,
        )

        assert new_access_token is not None
        assert returned_refresh == refresh_token

        # Verify new access token is stored
        new_access_key = OAUTH_ACCESS_TOKEN_REDIS_KEY.format(client_id=client_id, token=new_access_token)
        assert redis_client.get(new_access_key) is not None

        # Cleanup
        redis_client.delete(new_access_key)
        redis_client.delete(OAUTH_REFRESH_TOKEN_REDIS_KEY.format(client_id=client_id, token=refresh_token))

    def test_sign_access_token_returns_none_for_unknown_grant_type(self, flask_req_ctx_with_containers):
        grant_type = cast(OAuthGrantType, "invalid-grant-type")

        result = OAuthServerService.sign_oauth_access_token(grant_type=grant_type, client_id=f"client-{uuid4()}")

        assert result is None

    def test_validate_access_token_returns_none_when_not_found(self, flask_req_ctx_with_containers):
        result = OAuthServerService.validate_oauth_access_token(f"client-{uuid4()}", f"missing-{uuid4()}")

        assert result is None

    def test_validate_access_token_loads_user_when_exists(self, flask_req_ctx_with_containers, db_session_with_containers):
        client_id = f"client-{uuid4()}"
        user_id = f"user-{uuid4()}"

        # Create tokens via the full flow
        code = OAuthServerService.sign_oauth_authorization_code(client_id, user_id)
        access_token, refresh_token = OAuthServerService.sign_oauth_access_token(
            grant_type=OAuthGrantType.AUTHORIZATION_CODE,
            code=code,
            client_id=client_id,
        )

        # Validate — AccountService.load_user needs to be mocked since we don't have a real account
        from unittest.mock import MagicMock

        expected_user = MagicMock()
        with patch("services.oauth_server.AccountService.load_user", return_value=expected_user) as mock_load:
            result = OAuthServerService.validate_oauth_access_token(client_id, access_token)

        assert result is expected_user
        mock_load.assert_called_once_with(user_id)

        # Cleanup
        redis_client.delete(OAUTH_ACCESS_TOKEN_REDIS_KEY.format(client_id=client_id, token=access_token))
        redis_client.delete(OAUTH_REFRESH_TOKEN_REDIS_KEY.format(client_id=client_id, token=refresh_token))

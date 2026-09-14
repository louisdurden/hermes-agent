from urllib.parse import parse_qs, urlsplit

from pydantic import AnyHttpUrl

from tools.mcp_oauth_provider import repair_authorization_endpoint_query


def test_repairs_railway_authorization_endpoint_query():
    endpoint = (
        "https://backboard.railway.com/oauth/auth"
        "?resource=https%3A%2F%2Fbackboard.railway.com"
    )
    malformed = endpoint + (
        "?response_type=code&client_id=client&redirect_uri=http%3A%2F%2F127.0.0.1%3A27890%2Fcallback"
        "&state=generated&code_challenge=challenge&code_challenge_method=S256"
        "&resource=https%3A%2F%2Fmcp.railway.com"
    )

    repaired = repair_authorization_endpoint_query(endpoint, malformed)
    query = parse_qs(urlsplit(repaired).query)

    assert query["response_type"] == ["code"]
    assert query["resource"] == ["https://mcp.railway.com"]
    assert query["client_id"] == ["client"]
    assert query["redirect_uri"] == ["http://127.0.0.1:27890/callback"]
    assert query["state"] == ["generated"]
    assert query["code_challenge"] == ["challenge"]
    assert "?response_type" not in query["resource"][0]


def test_generated_oauth_parameters_replace_endpoint_collisions():
    endpoint = (
        "https://auth.example.com/authorize?state=hostile&response_type=token"
        "&keep=one&keep=two&blank="
    )
    malformed = endpoint + "?state=generated&response_type=code&client_id=client"

    repaired = repair_authorization_endpoint_query(endpoint, malformed)
    query = parse_qs(urlsplit(repaired).query, keep_blank_values=True)

    assert query["state"] == ["generated"]
    assert query["response_type"] == ["code"]
    assert query["keep"] == ["one", "two"]
    assert query["blank"] == [""]


def test_leaves_normal_and_future_fixed_urls_unchanged():
    endpoint = "https://auth.example.com/authorize"
    normal = endpoint + "?response_type=code"
    assert repair_authorization_endpoint_query(endpoint, normal) == normal

    queried_endpoint = endpoint + "?resource=server"
    future_fixed = endpoint + "?resource=server&response_type=code"
    assert repair_authorization_endpoint_query(queried_endpoint, future_fixed) == future_fixed


def test_accepts_pydantic_authorization_endpoint():
    endpoint = AnyHttpUrl(
        "https://backboard.railway.com/oauth/auth"
        "?resource=https%3A%2F%2Fbackboard.railway.com"
    )
    malformed = str(endpoint) + "?response_type=code&client_id=client"

    repaired = repair_authorization_endpoint_query(endpoint, malformed)

    assert parse_qs(urlsplit(repaired).query)["response_type"] == ["code"]

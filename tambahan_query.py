curl -v https://llmqwen14b.wssdx.apps.cml.ocbcnisp.com/v1/models

import httpx
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    http_client=httpx.Client(verify=False)  # matikan verifikasi SSL sementara utk tes
)

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
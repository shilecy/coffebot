import requests
import json

resp = requests.get("http://localhost:8000/openapi.json")
with open("openapi_spec.json", "w") as f:
    json.dump(resp.json(), f, indent=2)

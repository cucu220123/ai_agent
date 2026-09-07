from app.llm.api_router import APIRoleRouter


class Backend:
    def __init__(self, name):
        self.model = name
        self.last_provider = "test_api"
        self.last_usage = {"total_tokens": 7}
        self.last_generation = {"finish_reason": "stop"}
        self.last_retry_count = 1

    def complete(self, *args, **kwargs):
        return self.model


def test_codegen_and_repair_route_to_coder_with_actual_usage_and_model():
    router = APIRoleRouter(Backend("instruction"), Backend("coder"))
    for purpose, expected in (("requirement", "instruction"), ("planning", "instruction"), ("code_generation", "coder"), ("repair", "coder"), ("explanation", "instruction")):
        assert router.complete("system", "request", purpose) == expected
        assert router.model == expected
        assert router.last_usage == {"total_tokens": 7}
        assert router.last_retry_count == 1

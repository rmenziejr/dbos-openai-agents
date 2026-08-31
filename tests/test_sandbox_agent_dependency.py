from agents.sandbox import SandboxAgent


def test_agents_sdk_exposes_sandbox_agent() -> None:
    assert SandboxAgent.__name__ == "SandboxAgent"

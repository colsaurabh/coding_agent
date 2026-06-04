import docker

class DockerSandboxController:
    def __init__(self, container_name="agent_sandbox"):
        try:
            self.client = docker.from_env()
            self.container_name = container_name
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Docker Daemon. Error: {e}")

    def run_command(self, command: str) -> dict:
        try:
            container = self.client.containers.get(self.container_name)
            
            # CRITICAL FIX: Wrap the command string explicitly inside a bash shell context
            # to let redirection operators like >, <, and << work seamlessly.
            wrapped_command = ["/bin/bash", "-c", command]
            
            exec_result = container.exec_run(
                cmd=wrapped_command,
                workdir="/workspace",
                demux=True
            )
            stdout = exec_result.output[0].decode('utf-8') if exec_result.output[0] else ""
            stderr = exec_result.output[1].decode('utf-8') if exec_result.output[1] else ""
            return {"exit_code": exec_result.exit_code, "stdout": stdout, "stderr": stderr}
        except docker.errors.NotFound:
            return {"exit_code": -1, "stdout": "", "stderr": f"Sandbox container '{self.container_name}' is not running."}
        except Exception as e:
            return {"exit_code": -99, "stdout": "", "stderr": str(e)}

    def run_compile_check(self) -> str:
        """Runs the workspace compilation verification check or installs dependencies."""
        print("🔧 Tool Call -> Compiling/Validating workspace application...")
        res = self.run_command("agent-build")
        return f"COMPILE STATUS CODE: {res['exit_code']}\nSTDOUT:\n{res['stdout']}\nSTDERR:\n{res['stderr']}"

    def run_test_suite(self) -> str:
        """Executes the test runner suite to evaluate code correctness and business behavior."""
        print("🔧 Tool Call -> Running unit/integration test suites...")
        res = self.run_command("agent-test")
        return f"TEST STATUS CODE: {res['exit_code']}\nSTDOUT:\n{res['stdout']}\nSTDERR:\n{res['stderr']}"

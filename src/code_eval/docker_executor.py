"""
Docker-based secure code execution for HumanEval evaluation.
"""

import os
import tempfile
import subprocess
import json
from typing import Dict, Any, Optional
from pathlib import Path


class DockerExecutor:
    """Executes Python code securely in a Docker container."""

    def __init__(
        self,
        image_name: str = "humaneval-sandbox:latest",
        timeout: float = 5.0,
        memory_limit: str = "512m",
        cpu_quota: int = 50000,  # 50% of one CPU core
    ):
        """
        Initialize the Docker executor.

        Args:
            image_name: Name of the Docker image to use
            timeout: Maximum execution time in seconds
            memory_limit: Memory limit for container (e.g., "512m", "1g")
            cpu_quota: CPU quota in microseconds per 100ms period
        """
        self.image_name = image_name
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self._ensure_image_built()

    def _ensure_image_built(self):
        """Build the Docker image if it doesn't exist."""
        result = subprocess.run(
            ["docker", "images", "-q", self.image_name],
            capture_output=True,
            text=True,
        )

        if not result.stdout.strip():
            print(f"Building Docker image: {self.image_name}")
            dockerfile_path = Path(__file__).parent / "Dockerfile"
            subprocess.run(
                ["docker", "build", "-t", self.image_name, "-f", str(dockerfile_path), "."],
                cwd=Path(__file__).parent,
                check=True,
            )
            print(f"Docker image {self.image_name} built successfully")

    def execute_code(
        self,
        code: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute Python code in a Docker container.

        Args:
            code: Python code to execute
            timeout: Override default timeout for this execution

        Returns:
            Dict with keys:
                - passed: bool, whether execution succeeded
                - result: str, "passed", "timed out", or error message
                - stdout: str, captured stdout
                - stderr: str, captured stderr
        """
        timeout = timeout or self.timeout

        # Create temporary file with code
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Run code in Docker container
            result = subprocess.run(
                [
                    "docker", "run",
                    "--rm",
                    "--network", "none",  # Disable network access
                    "--memory", self.memory_limit,
                    "--cpu-quota", str(self.cpu_quota),
                    "--cpus", "1",
                    "-v", f"{temp_file}:/sandbox/code.py:ro",  # Mount as read-only
                    "--security-opt", "no-new-privileges",
                    "--cap-drop", "ALL",  # Drop all capabilities
                    "--pids-limit", "50",  # Limit number of processes
                    self.image_name,
                    "/sandbox/code.py",
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 1,  # Add 1 second buffer to Docker timeout
            )

            # Check if execution passed (exit code 0)
            if result.returncode == 0:
                return {
                    "passed": True,
                    "result": "passed",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                return {
                    "passed": False,
                    "result": f"failed: exit code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

        except subprocess.TimeoutExpired:
            # Kill the container if it's still running
            return {
                "passed": False,
                "result": "timed out",
                "stdout": "",
                "stderr": f"Execution exceeded {timeout} seconds",
            }
        except Exception as e:
            return {
                "passed": False,
                "result": f"failed: {str(e)}",
                "stdout": "",
                "stderr": str(e),
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def check_correctness(
        self,
        task_id: str,
        prompt: str,
        completion: str,
        test: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Check if a completion is correct by running tests.

        Args:
            task_id: Identifier for the task
            prompt: The problem prompt/signature
            completion: The LLM-generated solution
            test: Test cases to verify the solution
            timeout: Override default timeout

        Returns:
            Dict with execution results including task_id
        """
        # Construct the full program
        program = f"{prompt}\n{completion}\n\n{test}\n\ncheck(candidate)"

        # Execute the program
        result = self.execute_code(program, timeout)
        result["task_id"] = task_id
        result["completion"] = completion

        return result


if __name__ == "__main__":
    # Test the executor
    executor = DockerExecutor()

    # Simple test
    test_code = """
def add(a, b):
    return a + b

assert add(2, 3) == 5
print("Test passed!")
"""

    result = executor.execute_code(test_code)
    print("Test result:", json.dumps(result, indent=2))

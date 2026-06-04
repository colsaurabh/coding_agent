import string
import random
from agent.tools.docker_exec import DockerSandboxController

class GitController:
    def __init__(self):
        self.sandbox = DockerSandboxController()

    def _sanitize_branch_name(self, text: str) -> str:
        allowed_chars = string.ascii_letters + string.digits + "-_"
        clean_text = "".join(c if c in allowed_chars else "-" for c in text).lower()
        clean_branch = "-".join(filter(None, clean_text.split("-")))[:30]
        random_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return f"fix/{clean_branch}-{random_id}"

    def initialize_feature_branch(self, issue_description: str) -> str:
        branch_name = self._sanitize_branch_name(issue_description)
        self.sandbox.run_command("git config --global user.email 'saurabhalld1989@gmail.com'")
        self.sandbox.run_command("git config --global user.name 'Saurabh Gupta'")
        self.sandbox.run_command("mkdir -p ~/.ssh")
        self.sandbox.run_command("ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts 2>/dev/null")
        self.sandbox.run_command("git checkout main")
        res = self.sandbox.run_command(f"git checkout -b {branch_name}")
        if res["exit_code"] == 0:
            return f"Successfully created and switched to local branch: {branch_name}"
        return f"Failed to initialize branch. Error:\n{res['stderr'] if res['stderr'] else res['stdout']}"

    def commit_verified_changes(self, commit_message: str) -> str:
        stage_res = self.sandbox.run_command("git add .")
        if stage_res["exit_code"] != 0:
            return f"Failed to stage files: {stage_res['stderr']}"
        escaped_msg = commit_message.replace("'", "'\\''")
        commit_res = self.sandbox.run_command(f"git commit -m '{escaped_msg}'")
        if commit_res["exit_code"] == 0:
            return f"Changes successfully committed locally:\n{commit_res['stdout']}"
        return f"Failed to create commit.\n{commit_res['stderr'] if commit_res['stderr'] else commit_res['stdout']}"

    def get_git_status(self) -> str:
        res = self.sandbox.run_command("git status --short")
        return res["stdout"] if res["exit_code"] == 0 else f"Error: {res['stderr']}"

    def push_feature_branch(self) -> str:
        """Pushes the current verified feature branch and its local commits to the remote origin server."""
        print("🔧 Tool Call -> Pushing committed branch to remote repository...")
        
        # 1. Get the name of the current active branch inside the container
        branch_res = self.sandbox.run_command("git branch --show-current")
        branch_name = branch_res["stdout"].strip()
        
        if not branch_name:
            return "Error: Could not determine current active git branch name."
            
        # 2. Push the branch to origin
        # Note: We use --set-upstream (or -u) so the branch tracks cleanly on the remote server
        push_res = self.sandbox.run_command(f"git push origin {branch_name}")
        
        if push_res["exit_code"] == 0:
            return f"Successfully pushed branch '{branch_name}' to remote origin repository.\n{push_res['stdout']}"
        
        # If it fails, capture the error output (likely an authentication error)
        return f"Failed to push branch to remote origin. Diagnostic info:\nSTDOUT:\n{push_res['stdout']}\nSTDERR:\n{push_res['stderr']}"

import os
import json
from google import genai
from google.genai import types

from agent.knowledge.pdf_rag import FunctionalKnowledgeEngine
from agent.knowledge.code_indexer import CodebaseIndexer
from agent.tools.docker_exec import DockerSandboxController
from agent.tools.git_control import GitController

knowledge_engine = FunctionalKnowledgeEngine()
sandbox = DockerSandboxController()
git_control = GitController()
client = genai.Client()

def read_file_content(file_path: str) -> str:
    result = sandbox.run_command(f"cat {file_path}")
    if result["exit_code"] == 0: return result["stdout"]
    return f"Error reading file: {result['stderr']}"

def write_file_content(file_path: str, content: str) -> str:
    """Overwrites or creates a file with new code content in the workspace."""
    print(f"🔧 Tool Call -> Writing file: {file_path}")
    
    # Base64 encode the string content to safely pass it into the container shell 
    # without escaping special characters like quotes or brackets.
    import base64
    b64_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    
    cmd = f"echo '{b64_content}' | base64 -d > {file_path}"
    result = sandbox.run_command(cmd)
    
    if result["exit_code"] == 0:
        return f"Successfully updated {file_path}"
    return f"Error writing file: {result['stderr']}"

def check_code_compilation() -> str:
    """Compiles the application codebase to check for any syntax errors or build issues."""
    return sandbox.run_compile_check()

def execute_repository_tests() -> str:
    """Runs the testing framework across the workspace to see if code changes behave properly."""
    return sandbox.run_test_suite()

def search_workspace_directory() -> str:
    res = sandbox.run_command("find . -maxdepth 3 -not -path '*/.*' -not -path '*/obj/*' -not -path '*/bin/*'")
    return res["stdout"]

agent_tools = [
    read_file_content,
    write_file_content,
    check_code_compilation,
    execute_repository_tests,
    search_workspace_directory,
    git_control.initialize_feature_branch,
    git_control.commit_verified_changes,
    git_control.get_git_status,
    git_control.push_feature_branch
]

def solve_issue(issue_description: str):
    print(f"🚀 Starting Autonomous Run for Issue: '{issue_description}'")
    business_rules_context = knowledge_engine.query_business_rules(issue_description, limit=3)

    # relevant_code_snippet = indexer.query_relevant_code(issue_description, top_k=50)
    relevant_code_snippet = "TODO: Integrate code retrieval from vector DB here. Currently hardcoded for testing."
    
    system_instruction = f"""
    You are an expert autonomous senior .NET software engineering agent. 
    Your primary goal is to resolve functional and technical bugs in a .NET application.

    RELEVANT CODE SNIPPETS RETRIEVED VIA RAG:
    {relevant_code_snippet}
    
    CRITICAL BUSINESS CONTEXT REPRODUCED FROM PRDs/DOCUMENTS:
    {business_rules_context if business_rules_context else "No direct matching business rules found."}
    
    YOUR RULES OF ENGAGEMENT:
    1. BEFORE modifying any code, call `initialize_feature_branch` to isolate your work environment.
    2. Scan the directory to identify the language stack (e.g., look for .cs, .java, .py, pom.xml, or csproj files) and source files.
    3. Read files to locate the functional logic bug or compilation failure point.
    4. Modify code using `write_file_content`.
    5. Run `check_code_compilation` to guarantee the change is syntactically sound.
    6. Run `execute_repository_tests` to verify functional behavior against business rules.
    7. If builds or tests fail, analyze the logs, rethink your logic, and issue a new fix.
    8. Once all tests pass perfectly, invoke `commit_verified_changes` with a summary of what you fixed to secure your solution locally.
    9. IMMEDIATELY after a successful commit, invoke `push_feature_branch` to deliver your solution back to the central repository. Do not conclude your run until the push is verified.
    """
    
    # Pass functions directly to tools instead of creating manual declaration lists.
    # The google-genai SDK handles mapping the parameters automatically this way.
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=agent_tools,
        temperature=0.2
    )
    
    chat = client.chats.create(model="gemini-3-flash-preview", config=config)
    current_prompt = f"Please analyze and resolve this issue: {issue_description}"
    
    for loop_idx in range(10):
        response = chat.send_message(current_prompt)
        # Check if the model wants to run a tool
        if response.function_calls:
            for call in response.function_calls:
                # Dynamically look up the matching Python tool function
                tool_func = next((f for f in agent_tools if f.__name__ == call.name), None)
                if tool_func:
                    tool_args = dict(call.args) if call.args else {}
                    
                    print(f"\n========================================================")
                    print(f"⚙️  GEMINI REQUESTED TOOL: [{call.name}]")
                    print(f"📦 Arguments Passed: {json.dumps(tool_args, indent=2)}")
                    print(f"========================================================")
                    
                    try:
                        # Execute the physical backend/Docker command
                        print("⏳ Executing command inside the sandbox container... Please wait...")
                        tool_output = tool_func(**tool_args)
                        
                        # NEW VISIBILITY LOGS: Print exactly what happened inside the sandbox
                        print(f"\n📥 [SANDBOX OUTPUT RECEIVED - {len(str(tool_output))} chars]")
                        print("--------------------------------------------------------")
                        print(tool_output) # This streams the RAW build errors, test failures, or file content to your terminal
                        print("--------------------------------------------------------\n")
                        
                    except TypeError as e:
                        print(f"⚠️ Argument mapping error encountered: {e}. Attempting recovery...")
                        if len(tool_args) == 1 and 'file_path' in tool_args:
                            tool_output = tool_func(tool_args['file_path'])
                        else:
                            tool_output = f"TypeError during tool execution: {str(e)}."
                    
                    # Feed the outcome back to Gemini on the next cycle
                    current_prompt = f"Tool '{call.name}' returned output:\n{tool_output}"
                else:
                    current_prompt = f"Error: Tool '{call.name}' not recognized by supervisor system."
        else:
            print("🏁 Final Resolution:\n", response.text)
            break

if __name__ == "__main__":

    # indexer = CodebaseIndexer(repo_path="/workspace")
    # indexer = CodebaseIndexer(repo_path="/Users/saurabhgupta/Saurabh/L&T Finance/Coding/SamplePaymentService/DotNet/PaymentService")
    # indexer.index_repository()

    # solve_issue("Find the payment processing controller and verify if the currency validation logic is correct.")
    solve_issue(
        "Create a dummy file named 'git_test_1.txt' containing the text 'Docker Git Pipeline Test'. "
        "Initialize a feature branch, add this file, commit the change, and run the push tool to verify remote connectivity."
    )
# Agent Environment

The agents run in a Docker image which is based off of `python:3.11-slim`. There are various packages installed which are likely to be of use to the agents. These are outlined in `/testbed/sandbox/sandbox_requirements.txt`. See `/testbed/sandbox/Dockerfile` for more information.

Here is the directory structure that an agent can expect to see. This is unified across *all* problem sets.

```
/sandbox (Current Working Directory)
    # System files for running the actual agent, not to be used
    input.json
    output.json
    AGENT_RUNNER.py

    # The actual agent source code
    agent.py



    # The repository that the agent should work in (has .git/, etc.)
    repo/
        # If include_solution (--include-solution) is set
        solution.diff

        # The contents of this directory are specific to the suite and problem
```

Here is the how the agent should be structured.

```py
def agent_main(input):
    problem_statement = input.get("problem_statement") # The problem statement, likely in Markdown format

    # Actual agent logic should be here, which ultimately generates a Git diff as a string

    return diff;
```

You also have some environment variables exposed.

```bash
RUN_ID # Unique UUID for this given evaluation run
SANDBOX_PROXY_URL # URL (such as http://sandbox_proxy:80) that can be used for making inference/embedding requests
```

The agents can also make requests for inference and embedding. Sample functions for both are provided in `test_agent.py`.
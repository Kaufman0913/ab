import os
import sys
import json
import time
import uuid
import argparse

from sandbox.sandbox_manager import SandboxManager
from problem_suites.polyglot.polyglot_suite import PolyglotSuite
from utils.logger import info, warn, error, debug, enable_verbose
from problem_suites.swebench_verified.swebench_verified_suite import SWEBenchVerifiedSuite



POLYGLOT_PROBLEMS = [
    "affine-cipher", "beer-song", "bowling", "connect", "dot-dsl",
    "food-chain", "forth", "go-counting", "grade-school", "hangman",
    "list-ops", "phone-number", "pig-latin", "poker", "pov", "proverb",
    "react", "rest-api", "robot-name", "scale-generator", "simple-linked-list",
    "transpose", "variable-length-quantity", "wordy", "zebra-puzzle"
]

SWEBENCH_VERIFIED_PROBLEMS = [
    "astropy__astropy-13398", "astropy__astropy-13579", "django__django-11138",
    "django__django-11400", "django__django-12325", "django__django-12708",
    "django__django-13128", "django__django-13212", "django__django-13449",
    "django__django-13837", "django__django-14007", "django__django-14011",
    "django__django-14631", "django__django-15268", "django__django-15503",
    "django__django-15629", "django__django-15957", "django__django-16263",
    "django__django-16560", "django__django-16631", "pytest-dev__pytest-5787",
    "pytest-dev__pytest-6197", "pytest-dev__pytest-10356",
    "sphinx-doc__sphinx-9461", "sphinx-doc__sphinx-11510"
]

AGENT_PATH = "test_agent.py"

AGENT_TIMEOUT = 600
EVAL_TIMEOUT = 600

INCLUDE_SOLUTION = True

RUNS_DIR = "runs2"



def run_problem_from_suite(suite, sandbox_manager, problem_name, agent_source_code):
    run_id = str(uuid.uuid4())



    result_data = {}
    
    def on_agent_finish(agent_result):
        result_data["agent_result"] = agent_result

        if agent_result["status"] == "success":
            def on_eval_finish(eval_result):
                if eval_result["status"] == "success":
                    result_data["eval_result"] = eval_result
                else:
                    warn(f"Evaluation failed on {problem_name}")

                output_file = os.path.join(RUNS_DIR, f"{problem_name}.json")
                with open(output_file, "w") as f:
                    json.dump(result_data, f, indent=2)
                info(f"{problem_name} ----> {output_file}")

            suite.evaluate_solution_diff(
                sandbox_manager,
                run_id,
                problem_name,
                agent_result["diff"],
                on_eval_finish,
                timeout=EVAL_TIMEOUT
            )
        else:
            warn(f"Agent failed on {problem_name}")

            output_file = os.path.join(RUNS_DIR, f"{problem_name}.json")
            with open(output_file, "w") as f:
                json.dump(result_data, f, indent=2)
            info(f"{problem_name} ----> {output_file}")
    


    suite.run_agent_in_sandbox_for_problem(
        sandbox_manager,
        run_id,
        problem_name,
        agent_source_code,
        on_agent_finish,
        timeout=AGENT_TIMEOUT,
        include_solution=INCLUDE_SOLUTION
    )


def run_benchmark(gateway_url):
    start_time = time.time()



    polyglot_suite = PolyglotSuite("./datasets/polyglot")
    swebench_verified_suite = SWEBenchVerifiedSuite("./datasets/swebench_verified")

    sandbox_manager = SandboxManager(gateway_url)

    swebench_verified_suite.prebuild_problem_images(sandbox_manager, SWEBENCH_VERIFIED_PROBLEMS)



    with open(AGENT_PATH, "r") as f:
        agent_source_code = f.read()



    for problem_name in POLYGLOT_PROBLEMS:
        run_problem_from_suite(polyglot_suite, sandbox_manager, problem_name, agent_source_code)

    for problem_name in SWEBENCH_VERIFIED_PROBLEMS:
        run_problem_from_suite(swebench_verified_suite, sandbox_manager, problem_name, agent_source_code)



    while sandbox_manager.get_num_sandboxes() > 0:
        time.sleep(1)



    elapsed_time = time.time() - start_time
    info(f"Time: {elapsed_time:.1f} seconds")

    return 0



def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runner for problem suite evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python benchmark.py http://localhost:8000"""
    )
    
    parser.add_argument("gateway_url", help="URL for the gateway")
    
    args = parser.parse_args()
    
    return run_benchmark(args.gateway_url)



if __name__ == "__main__":
    sys.exit(main())
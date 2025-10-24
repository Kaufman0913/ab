"""Agent runner CLI for problem suite benchmarks."""

import os
import sys
import time
import uuid
import argparse

from sandbox.sandbox_manager import SandboxManager
from problem_suites.polyglot.polyglot_suite import PolyglotSuite
from utils.logger import info, warn, error, debug, enable_verbose
from problem_suites.swebench_verified.swebench_verified_suite import SWEBenchVerifiedSuite



def run_agent_on_problem(suite_name, problem_name, agent_file, gateway_url, *, log_docker_to_stdout=False, include_solution=False, timeout=300):
    suite_configs = {
        "polyglot": {
            "class": PolyglotSuite,
            "path": "datasets/polyglot"
        },
        "swebench_verified": {
            "class": SWEBenchVerifiedSuite,
            "path": "datasets/swebench_verified"
        }
    }
    
    if suite_name not in suite_configs:
        error(f"Unknown suite: {suite_name}. Available suites: {list(suite_configs.keys())}")
        return 1

    config = suite_configs[suite_name]
    suite = config["class"](config["path"])
    
    test_count = suite.get_problem_test_count(problem_name)
    if test_count > 150:
        error(f"Problem {problem_name} has {test_count} tests (>150)")
        return 1
    
    info(f"Problem {problem_name} has {test_count} tests")
    


    sandbox_manager = SandboxManager(gateway_url, log_docker_to_stdout=log_docker_to_stdout)



    with open(agent_file, "r") as f:
        agent_source_code = f.read()

    run_id = str(uuid.uuid4())



    

    def on_finish(result):
        time.sleep(0.5)

        print()
        print()
        print()

        if (result["status"] == "success"):
            n = len((result.get("diff") or "").splitlines())
            print(f"========== DIFF ({n} line{'s' if n != 1 else ''}) ==========")
            print(result.get("diff", ""))

            n = len((result.get("logs") or "").splitlines())
            print(f"========== LOGS ({n} line{'s' if n != 1 else ''}) ==========")
            # print(result.get("logs", ""))

            print()
            print()
            print()
            
            diff = result["diff"]
            
            def on_finish(result):
                time.sleep(0.5)
                
                print()
                print()
                print()

                if result["status"] == "success":
                    print("========== TEST RESULTS ==========")
                    test_results = result.get("test_results", [])
                    tests_passed = sum(1 for test in test_results if test["status"] == "pass")
                    tests_failed = sum(1 for test in test_results if test["status"] == "fail")
                    tests_skipped = sum(1 for test in test_results if test["status"] == "skip")
                    print(f"{tests_passed} passed, {tests_failed} failed, {tests_skipped} skipped")
                    for test in test_results:
                        print(f"{test['name']} - {test.get('category', 'no category')} - {test['status']}")

                    n = len((result.get("logs") or "").splitlines())
                    print(f"========== LOGS ({n} line{'s' if n != 1 else ''}) ==========")
                    # print(result["logs"])
                else:
                    print("========== ERROR ==========")
                    print(result.get("error", ""))
                    
                    print("========== TRACEBACK ==========")
                    print(result.get("traceback", ""))

                    print("========== LOGS ==========")
                    print(result.get("logs", ""))

                print()
                print()
                print()
            
            suite.evaluate_solution_diff(sandbox_manager, run_id, problem_name, diff, on_finish, timeout=timeout)
        else:
            print("========== ERROR ==========")
            print(result.get("error", ""))

            print("========== TRACEBACK ==========")
            print(result.get("traceback", ""))

            print("========== DIFF ==========")
            print(result.get("diff", ""))

            print("========== LOGS ==========")
            print(result.get("logs", ""))
    


    suite.run_agent_in_sandbox_for_problem(sandbox_manager, run_id, problem_name, agent_source_code, on_finish, timeout=timeout, include_solution=include_solution)
    


    time.sleep(1)
    while sandbox_manager.get_num_sandboxes() > 0:
        time.sleep(1)
    


    return 0



def main():
    parser = argparse.ArgumentParser(
        description="Agent runner CLI for problem suite benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python cli.py polyglot affine-cipher test_agent.py http://localhost:8000
  python cli.py swebench_verified django__django-12308 test_agent.py http://localhost:8000
  python cli.py polyglot affine-cipher test_agent.py http://localhost:8000 --include-solution --log-docker-to-stdout --verbose"""
    )
    
    parser.add_argument("suite_name", help="Problem suite name (polyglot, swebench_verified)")
    parser.add_argument("problem_name", help="Name of the specific problem to run")
    parser.add_argument("agent_file", help="Path to the agent Python file")
    parser.add_argument("gateway_url", help="URL for the gateway")
    
    parser.add_argument("--log-docker-to-stdout", action="store_true", 
                       help="Print Docker container logs to stdout in real-time")
    parser.add_argument("--include-solution", action="store_true",
                       help="Expose the solution to the agent at /sandbox/solution.diff")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose (debug) logging")
    parser.add_argument("--timeout", type=int, default=10,
                       help="Timeout in seconds for sandbox execution (default: 10)")
    
    args = parser.parse_args()
    
    if args.verbose:
        enable_verbose()
    
    return run_agent_on_problem(
        args.suite_name, 
        args.problem_name, 
        args.agent_file,
        args.gateway_url,
        log_docker_to_stdout=args.log_docker_to_stdout,
        include_solution=args.include_solution,
        timeout=args.timeout
    )



if __name__ == "__main__":
    sys.exit(main())
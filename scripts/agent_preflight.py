import argparse
from pathlib import Path

from web.agent import AgentError, PreflightSession
from web.configs import AgentConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Codex chat against HEPTAPOD preflight without running inspection.")
    parser.add_argument('--config-path', default='configs/agent_preflight.yaml')
    parser.add_argument('--check-connection', action='store_true', help="Discover the preflight tool without starting a model turn.")
    parser.add_argument('--message', default="Check this board's readiness and explain any blockers. Do not repair anything.")
    parser.add_argument('--follow-up', help="Optional second message in the same conversation; uses captured evidence.")
    args = parser.parse_args()
    session = None
    try:
        config = AgentConfig.from_yaml(Path(args.config_path))
        session = PreflightSession(config)
        with session:
            print(f"Connected to Codex and HEPTAPOD. Evidence directory: {session.run_dir}", flush=True)
            if args.check_connection:
                return 0

            response = session.ask(args.message)
            print(response['answer'], flush=True)
            ready = session.report['preflight']['result']['ready']
            print(f"Verified tool readiness: {ready}")
            if args.follow_up:
                print(session.ask(args.follow_up, require_preflight=False)['answer'])

            return 0 if ready else 2
    except KeyboardInterrupt:
        print("Cancelled; completed diagnostic artifacts have been retained.")
        return 130
    except (AgentError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"Preflight chat failed: {error}")
        if session is not None:
            print(f"Diagnostics: {session.run_dir}")

        return 1


if __name__ == '__main__':
    raise SystemExit(main())

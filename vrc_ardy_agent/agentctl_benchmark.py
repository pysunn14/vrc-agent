"""Explicit opt-in real API evaluation, with offline defaults."""
import json
from pathlib import Path
import time


def add_benchmark_arguments(groups):
    parser = groups.add_parser('benchmark', help='Record-only observation policy evaluation')
    commands = parser.add_subparsers(dest='benchmark_command',required=True)
    run = commands.add_parser('run')
    run.add_argument('--cases',type=Path,default=Path(__file__).resolve().parents[1]/'benchmarks/observation/cases.json')
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--real-api',action='store_true')
    run.add_argument('--rubric-approved',action='store_true')
    run.add_argument('--max-calls',type=int)
    run.add_argument('--repeats',type=int,default=3)
    run.add_argument('--seed',type=int,default=20260915)
    run.add_argument('--limit',type=int)
    run.add_argument('--timeout',type=float,default=30.)
    run.add_argument('--hermes-env-file',type=Path,default=Path.home()/'.hermes/.env')
    for name in ('status','report'):
        command=commands.add_parser(name)
        command.add_argument('--output',type=Path,required=True)


def run_benchmark_command(args):
    from .benchmark_report import summarize, turn_records
    from .observation_benchmark import run_benchmark
    if args.benchmark_command=='report': return summarize(args.output)
    if args.benchmark_command=='status':
        manifest,turns=turn_records(args.output)
        return dict(run_id=manifest['run_id'],mode=manifest['mode'],total=len(manifest['schedule']),
                    completed=sum(t['status'] not in ('interrupted',) for t in turns),
                    started=len(turns),pending_calls=sum(t['pending_calls'] for t in turns))
    brain=None
    if args.real_api:
        if not args.rubric_approved or args.max_calls is None:
            raise ValueError('real API requires --rubric-approved and --max-calls')
        from .hermes_brain import HermesBrainAdapter, load_hermes_environment
        brain=HermesBrainAdapter.from_environment(load_hermes_environment(args.hermes_env_file))
    def progress(row):
        print(json.dumps(dict(timestamp_unix=time.time(),**row),ensure_ascii=False),flush=True)
    result=run_benchmark(cases_path=args.cases,output=args.output,repeats=args.repeats,
        seed=args.seed,real_api=args.real_api,max_calls=args.max_calls,rubric_approved=args.rubric_approved,
        limit=args.limit,timeout_seconds=args.timeout,brain=brain,progress=progress)
    summarize(args.output)
    return result

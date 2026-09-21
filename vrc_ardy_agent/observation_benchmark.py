"""Policy-only planning experiments. No VRChat, audio or device dependencies."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import threading
import time
import uuid

from .action_contracts import ActionType
from .benchmark_store import EventStore, atomic_json, code_identity, digest, read_events
from .character_supervisor import CharacterSupervisor
from .hermes_brain import HermesBrainAdapter, _SYSTEM_PROMPT
from .interaction_runtime import InteractionRuntime
from .motion_stream import MotionDirectorSnapshot, MotionDirectorState, MotionHaltResult
from .observation_loop import ObservationLoop, ReplayFrame
from .planning_trace import PlanningTrace


class RecordOnlyMotion:
    def snapshot(self):
        return MotionDirectorSnapshot(state=MotionDirectorState.IDLE, running=False)

    def submit_cue(self, *args, **kwargs):
        raise AssertionError('benchmark must never dispatch a device action')

    def halt(self, *, reason):
        return MotionHaltResult(reason=reason, lease_token=0)


class ForbiddenExecutor:
    def run(self, *args):
        raise AssertionError('benchmark must never execute speech')


class DryRunBrain:
    """Plumbing validation only; deterministic answers are not model measurements."""
    def __init__(self, case): self.case = case

    def propose(self, request):
        if self.case['vision_requirement'] == 'required' and request.screenshot is None:
            return {'type':'observe_scene','query':request.transcript}
        return {'requires_visual':self.case['vision_requirement']=='required',
                'actions':[{'type':'say','text':'평가 도구 점검용 응답입니다.'}]}


def load_cases(path):
    path = Path(path)
    data = json.loads(path.read_text())
    ids = set()
    for case in data['cases']:
        if case['case_id'] in ids: raise ValueError('duplicate case_id')
        ids.add(case['case_id'])
        if case['category'] not in ('nonvisual','visual','ambiguous'):
            raise ValueError('invalid category')
        if case['vision_requirement'] not in ('required','none'):
            raise ValueError('invalid vision requirement')
        image = path.parent/case['image']
        if digest(image.read_bytes()) != case['image_sha256']:
            raise ValueError('fixture image hash mismatch')
        # This pilot supports only this explicit initial state, not arbitrary
        # replayed execution feedback that the harness cannot reproduce.
        if case['initial_state'] != dict(body='IDLE',speech='IDLE',current_action_ids=[],last_error=None,tracking='unknown'):
            raise ValueError('unsupported initial state')
        if case['source'] != 'replay': raise ValueError('fixture source must be replay')
    return data


def balanced_schedule(cases, *, repeats, seed):
    if repeats < 1: raise ValueError('repeats must be positive')
    rng = random.Random(seed)
    order = []
    for repeat in range(repeats):
        indices = list(range(len(cases)))
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            policies = ('always','selective') if (position+repeat)%2 == 0 else ('selective','always')
            for policy in policies:
                order.append(dict(case_id=cases[index]['case_id'],repeat=repeat,policy=policy,warmup=False))
    return order


def _manifest(*, cases_path, repeats, seed, real_api, max_calls, rubric_approved, limit,
              timeout_seconds, brain, root):
    data = load_cases(cases_path)
    order = balanced_schedule(data['cases'], repeats=repeats, seed=seed)
    if limit is not None:
        if limit < 1 or limit%2: raise ValueError('limit must be a positive even number for paired runs')
        order = order[:limit]
    if real_api:
        if not rubric_approved: raise ValueError('human rubric approval is required for real API runs')
        warmup_case = next(c['case_id'] for c in data['cases'] if c['category']=='visual')
        order = [dict(case_id=warmup_case,repeat=-1,policy=p,warmup=True)
                 for p in ('always','selective')] + order
        if max_calls is None or max_calls < len(order)*3:
            raise ValueError(f'explicit call budget must allow at least {len(order)*3} worst-case calls')
    settings = dict(model=brain.model,provider=brain.provider,reasoning_effort=brain.reasoning_effort,
                    adapter_timeout_seconds=brain.timeout_seconds) if brain else dict(model='scripted_fake',provider='none')
    return dict(schema_version=1, mode='real_api' if real_api else 'dry_run',
        settings=settings, repeats=repeats, seed=seed, schedule=order,
        fixture_manifest_sha256=digest(Path(cases_path).read_bytes()), cases=data['cases'],
        rubric_approved=rubric_approved, prompt_sha256=digest(_SYSTEM_PROMPT.encode()),
        code=code_identity(root), timeout_seconds=timeout_seconds,max_observations=2,
        max_calls=max_calls, max_possible_calls=len(order)*3, retries=0,
        image_size=[640,480], image_detail='high', evidence_source='replay',
        executor='record_only', cost_basis=None,
        exclusions='Only the two declared warm-up turns are excluded. All evaluation failures remain.',
        policy_difference='always acquires evidence before round 0; selective starts without evidence',
        upstream_call_accounting='Counts adapter HTTP calls; any server-internal retry/inference is not independently observable.')


def run_benchmark(*, cases_path, output, repeats=3, seed=20260915, real_api=False,
                  max_calls=None, rubric_approved=False, limit=None, timeout_seconds=30.0,
                  brain=None, progress=None):
    import math
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('timeout must be positive and finite')
    root = Path(__file__).resolve().parents[1]
    # Validate authorization before creating a result directory or contacting a model.
    if real_api and not rubric_approved: raise ValueError('human rubric approval required')
    if real_api and brain is None:
        brain = HermesBrainAdapter.from_environment()
    config = _manifest(cases_path=cases_path,repeats=repeats,seed=seed,real_api=real_api,
        max_calls=max_calls,rubric_approved=rubric_approved,limit=limit,
        timeout_seconds=timeout_seconds,brain=brain,root=root)
    store = EventStore(output)
    active_loop = None
    try:
        manifest_path = Path(output)/'manifest.json'
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            if {k:v for k,v in saved.items() if k not in ('run_id','created_utc')} != config:
                raise ValueError('manifest/config/code changed; use a separate output directory')
            manifest = saved
        else:
            manifest = dict(config,run_id=uuid.uuid4().hex,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            atomic_json(manifest_path,manifest)
        rows = read_events(output)
        # A started attempt is never automatically repeated: its remote call may
        # already have incurred usage. Report interrupted attempts explicitly.
        attempted = {r['schedule_index'] for r in rows if r['event']=='turn.input'}
        cases = {c['case_id']:c for c in manifest['cases']}
        total = len(manifest['schedule'])
        for index, item in enumerate(manifest['schedule']):
            if index in attempted: continue
            case = cases[item['case_id']]
            context = dict(run_id=manifest['run_id'],schedule_index=index,category=case['category'],**item)
            def sink(row, context=context): store.append(dict(context, **row))
            trace_holder = []
            def trace_factory(token):
                trace = PlanningTrace(token.turn_id,token.version,sink=sink)
                trace_holder.append(trace)
                return trace
            plans = []
            supervisor = CharacterSupervisor(executors={ActionType.SAY:ForbiddenExecutor()})
            runtime = InteractionRuntime(brain=brain if real_api else DryRunBrain(case),
                supervisor=supervisor,motion_director=RecordOnlyMotion(),
                trace_factory=trace_factory,plan_recorder=plans.append)
            image_path = Path(cases_path).parent/case['image']
            def replay(image_path=image_path, expected_hash=case['image_sha256']):
                data = image_path.read_bytes()
                if digest(data) != expected_hash: raise ValueError('fixture changed during run')
                return ReplayFrame(data)
            runtime.configure_observation(replay,policy=item['policy'],timeout_seconds=timeout_seconds)
            active_loop = runtime.observation_service
            if progress: progress(dict(event='progress',completed=len(attempted),total=total,**context))
            outcome = runtime.handle_utterance(transcript=case['transcript'],screenshot=None)
            # Preserve late completion/usage on disk before continuing or closing
            # the logger. Do not silently replace a timed-out call with a retry.
            drain_deadline = time.monotonic() + (brain.timeout_seconds+5 if real_api else 5)
            while active_loop.snapshot()['active_calls']:
                if time.monotonic() >= drain_deadline:
                    raise RuntimeError('remote call remains pending; keep run unresolved and inspect server')
                if progress: progress(dict(event='draining',schedule_index=index,active_calls=active_loop.snapshot()['active_calls']))
                time.sleep(.2)
            attempted.add(index)
            atomic_json(Path(output)/'checkpoint.json',dict(run_id=manifest['run_id'],attempted=sorted(attempted),total=total,
                                                         last_status=outcome.status.value,heartbeat_unix=time.time()))
            if progress: progress(dict(event='progress',completed=len(attempted),total=total,last_status=outcome.status.value))
        return dict(run_id=manifest['run_id'],attempted=len(attempted),total=total)
    finally:
        # Normally every worker has drained. Keeping the store alive on an
        # exceptional pending call lets a late callback still append evidence.
        if active_loop is None or not active_loop.snapshot()['active_calls']:
            store.close()

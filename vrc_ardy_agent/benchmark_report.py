"""Rebuild summaries from original events, including late calls and failed turns."""
import csv
import json
from pathlib import Path
import statistics

from .benchmark_store import atomic_json, read_events


def turn_records(directory):
    manifest = json.loads((Path(directory)/'manifest.json').read_text())
    turns = {}
    calls = {}
    for row in read_events(directory):
        index = row['schedule_index']
        if index not in turns:
            turns[index] = dict(schedule_index=index,case_id=row['case_id'],repeat=row['repeat'],
                policy=row['policy'],category=row['category'],warmup=row['warmup'],turn_id=row['turn_id'],
                status='interrupted',plan_ms=None,elapsed_ms=None,validation_ms=None,
                failure_stage='interrupted',plan=None,observations=[],decisions=[])
        turn = turns[index]
        if row['event']=='turn.finished':
            for key in ('status','plan_ms','elapsed_ms','validation_ms','failure_stage','plan','error'):
                turn[key] = row.get(key)
        elif row['event']=='image.acquired':
            turn['observations'].append(row)
        elif row['event']=='model.decision':
            turn['decisions'].append(row['decision'])
        if row['event'].startswith('call.'):
            call = calls.setdefault(row['call_id'],{})
            call.update(row)
    for index,turn in turns.items():
        turn['calls'] = [c for c in calls.values() if c['schedule_index']==index]
        turn['model_calls'] = sum(c.get('kind')=='model' and c.get('started_ns') is not None for c in turn['calls'])
        turn['image_model_calls'] = sum(c.get('kind')=='model' and c.get('started_ns') is not None and c.get('has_image',False) for c in turn['calls'])
        turn['image_requests'] = sum(c.get('kind')=='image' for c in turn['calls'])
        turn['pending_calls'] = sum(c.get('status')=='running' for c in turn['calls'])
    return manifest, sorted(turns.values(),key=lambda x:x['schedule_index'])


def _median(values): return statistics.median(values) if values else None


def summarize(directory):
    directory = Path(directory)
    manifest, turns = turn_records(directory)
    grade_path = directory/'grades.json'
    grades = json.loads(grade_path.read_text()) if grade_path.exists() else []
    grade_by_id = {g['turn_id']:g for g in grades}
    cases = {c['case_id']:c for c in manifest['cases']}
    groups = []
    for category in ('nonvisual','visual','ambiguous'):
        for policy in ('always','selective'):
            samples=[t for t in turns if not t['warmup'] and t['category']==category and t['policy']==policy]
            passed=[t for t in samples if t['status']=='PLANNED']
            graded=[grade_by_id[t['turn_id']] for t in samples if t['turn_id'] in grade_by_id]
            groups.append(dict(category=category,policy=policy,n=len(samples),success=len(passed),
                failed=len(samples)-len(passed),median_plan_ms=_median([t['plan_ms'] for t in passed]),
                model_calls=sum(t['model_calls'] for t in samples),
                image_model_calls=sum(t['image_model_calls'] for t in samples),
                observations=sum(len(t['observations']) for t in samples),
                unnecessary_observation_turns=sum(bool(t['observations']) and cases[t['case_id']]['vision_requirement']=='none' for t in samples),
                required_without_evidence=sum(not t['observations'] and cases[t['case_id']]['vision_requirement']=='required' for t in passed),
                correct=sum(g.get('correct') is True for g in graded),graded=len(graded),
                confirmed_observation_omissions=sum(g.get('unsupported_visual_claim') is True for g in graded),
                ambiguity_appropriate=sum(g.get('ambiguity_appropriate') is True for g in graded)))
    paired=[]
    for category in ('nonvisual','visual','ambiguous'):
        pairs={}
        for t in turns:
            if not t['warmup'] and t['category']==category and t['status']=='PLANNED':
                pairs.setdefault((t['case_id'],t['repeat']),{})[t['policy']]=t
        pairs=[p for p in pairs.values() if set(p)=={'always','selective'}]
        a=_median([p['always']['plan_ms'] for p in pairs]);b=_median([p['selective']['plan_ms'] for p in pairs])
        paired.append(dict(category=category,n_pairs=len(pairs),median_always_ms=a,median_selective_ms=b,
            median_paired_delta_ms=_median([p['selective']['plan_ms']-p['always']['plan_ms'] for p in pairs]),
            reduction_percent=None if not a else (a-b)/a*100))
    summary=dict(run_id=manifest['run_id'],mode=manifest['mode'],
        completed_turns=sum(t['status']!='interrupted' for t in turns),
        evaluation_turns=sum(not t['warmup'] for t in turns),
        expected_turns=len(manifest['schedule']),pending_calls=sum(t['pending_calls'] for t in turns),
        human_graded=sum(g.get('reviewer_type')=='human' for g in grades),
        agent_graded=sum(g.get('reviewer_type')=='agent' for g in grades),
        total_model_calls=sum(t['model_calls'] for t in turns),
        usage_available_calls=sum(c.get('kind')=='model' and c.get('usage') is not None for t in turns for c in t['calls']),
        groups=groups,paired=paired,cost=None)
    atomic_json(directory/'summary.json',summary)
    fields=['schedule_index','case_id','repeat','policy','category','warmup','turn_id','status','plan_ms','elapsed_ms','validation_ms','failure_stage','model_calls','image_model_calls','image_requests','pending_calls','plan']
    with (directory/'turns.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for t in turns: writer.writerow(dict(t,plan=json.dumps(t['plan'],ensure_ascii=False)))
    lines=['# 선택적 시각 관측 비교 결과','',f"실행 모드: {manifest['mode']}. 평가 {summary['evaluation_turns']}턴, 준비 호출 포함 완료 {summary['completed_turns']}/{summary['expected_turns']}턴.",
        f"모델 호출 {summary['total_model_calls']}회. 미완료 호출 {summary['pending_calls']}회. 사람 채점 {summary['human_graded']}개, 에이전트 예비 채점 {summary['agent_graded']}개.",
        '', '| 그룹 | 정책 | 성공/전체 | 판단 중앙값(초) | 전체/이미지 모델 호출 | 관측 불필요 사례의 관측 턴 | 정답/채점 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for g in groups:
        latency='미측정' if g['median_plan_ms'] is None else f"{g['median_plan_ms']/1000:.3f}"
        lines.append(f"| {g['category']} | {g['policy']} | {g['success']}/{g['n']} | {latency} | {g['model_calls']}/{g['image_model_calls']} | {g['unnecessary_observation_turns']} | {g['correct']}/{g['graded']} |")
    lines += ['', '## 같은 사례에서 두 정책 모두 성공한 비교','',
              '| 그룹 | 쌍 수 | 항상 관측 중앙값(초) | 선택적 관측 중앙값(초) | 중앙값 감소율 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for p in paired:
        if p['n_pairs']:
            lines.append(f"| {p['category']} | {p['n_pairs']} | {p['median_always_ms']/1000:.3f} | {p['median_selective_ms']/1000:.3f} | {p['reduction_percent']:.1f}% |")
    lines += ['', '음수 감소율은 선택적 관측이 더 느렸다는 뜻입니다. 실패는 위 표의 전체 수에 포함하며, 판단 시간은 성공한 요청만 계산합니다.',
        '', '## 해석 범위','',
        '- 직접 만든 고정 도형 그림과 20개 입력의 반복 실험입니다. 실제 사용자 발화 분포와 브이알챗 성능으로 일반화하지 않습니다.',
        '- 저장 그림 획득, 모든 모델 재호출, 파싱과 최종 계획 검증을 포함합니다. 음성 인식·합성, 모션 생성·재생, 실제 캡처·전송은 제외합니다.',
        '- 호출 시간은 네트워크·서버 대기·어댑터 처리를 포함합니다. 순수 추론 시간이나 첫 토큰 지연이 아닙니다.',
        '- API 서버 내부의 재시도·모델 호출은 직접 관측하지 못합니다. 여기서 호출 수는 어댑터가 시작한 HTTP 요청 수입니다.',
        '- 요금 기준이 확인되지 않아 비용을 금액으로 환산하지 않습니다. 제공되지 않은 사용량은 null로 남깁니다.',
        '- 같은 20개 사례의 반복은 독립된 60개 사례가 아닙니다. 품질 점수는 채점 주체와 검토 여부를 함께 보아야 합니다.',
        '- dry_run 결과는 도구 동작 점검이며 실제 모델의 지연이나 품질 측정값이 아닙니다.']
    (directory/'summary.ko.md').write_text('\n'.join(lines)+'\n')
    return summary

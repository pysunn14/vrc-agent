# 선택적 시각 관측 평가

같은 `InteractionRuntime`, 행동 계약, 루나 프롬프트에서 최초 관측 정책만 바꿉니다. `always`는 첫 판단 전에 그림을 얻고, `selective`는 텍스트로 시작해서 관측 요청 또는 시각적 근거 요구가 있을 때 그림을 얻습니다. 이후 재관측·취소·검증 경로는 공통입니다.

실제 브이알챗과 오디오를 실행하지 않습니다. 장치 실행기에는 명령을 전달하지 않고, 현재 턴의 권한이 유효한 검증된 계획만 기록합니다. 결과 상태 `PLANNED`는 발화나 동작 완료를 뜻하지 않습니다.

## 입력과 기준

[cases.json](cases.json)은 비시각적 10개, 시각적 6개, 모호하거나 정보가 부족한 4개 사례를 담습니다. [사전 채점안](rubric.ko.md)을 사람이 확인한 뒤 실제 모델 실험을 실행합니다. 직접 제작한 도형 그림이므로 실제 이용자의 이미지나 대화를 포함하지 않습니다.

`build_fixtures.py`는 그림과 제안 상태의 사례 파일을 다시 생성합니다. 재생성하면 해시와 사람 검토 상태를 다시 확인해야 합니다. 진행 중인 실험의 입력을 재생성하지 않습니다.

## 실행

프로젝트 의존성이 설치된 파이썬으로 실행합니다. 기본 실행은 외부 호출이 없는 도구 점검입니다.

```sh
python scripts/agentctl.py benchmark run --output _local/artifacts/observation-dry-run --limit 4
python scripts/agentctl.py benchmark status --output _local/artifacts/observation-dry-run
python scripts/agentctl.py benchmark report --output _local/artifacts/observation-dry-run
```

기존 루나 서버 연결과 인증 환경이 준비되고 사례·호출 상한을 승인한 경우에만 실제 실험을 실행합니다. 인증값은 기존 환경이나 `--hermes-env-file`에서 읽으며 산출물에 저장하지 않습니다.

```sh
python scripts/agentctl.py benchmark run \
  --output _local/artifacts/observation-pilot \
  --real-api --rubric-approved --max-calls 366
```

20개 사례를 두 정책으로 3회 반복한 120턴에 준비 2턴을 더합니다. 같은 사례의 두 정책을 인접하게 실행하되 선행 정책 수를 균형 있게 섞습니다. 준비 2턴은 요약 지연에서 제외하고 호출량과 원시 로그에는 남깁니다. 평가 실패는 제외하지 않습니다. 턴마다 초기 실행 상태와 대화 기록을 비우고 새로운 식별자를 발급합니다.

턴당 모델 호출은 최대 3회, 이미지 획득은 최대 2회, 총 판단 제한 시간은 기본 30초입니다. 평가 도구는 자동 재시도를 하지 않습니다. 서버 내부 호출과 재시도는 이 도구의 HTTP 요청 수와 구분해야 합니다.

## 계측과 재개

`events.jsonl`은 호출 요청·시작·완료·실패·대기 중단·늦은 완료, 관측, 모델 결정, 최종 검증을 추가 기록합니다. 전체 시간은 같은 프로세스의 단조 시계를 사용합니다. 제공되지 않은 API 사용량은 `null`입니다.

`plan_ms`는 런타임 입력 처리부터 최종 계획 검증까지입니다. 모델 호출과 후속 관측, 파싱·검증을 포함하며 음성 인식·합성, 모션 생성·재생을 제외합니다. 실패나 취소에는 성공 판단 시간을 기록하지 않습니다. 고정 그림에는 `source=replay`, `freshness_basis=fixed_fixture`, 이미지 해시가 붙으며 방금 촬영한 화면으로 표현하지 않습니다.

같은 명령과 결과 폴더로 재개하면 이미 시작한 턴은 재호출하지 않습니다. 도중에 종료된 턴은 `interrupted`로 남습니다. 원격 요청은 비용이 발생했을 수 있으므로 자동 재실행하지 않습니다. 코드·설정·입력 해시가 달라졌으면 별도 결과 폴더를 사용해야 합니다. 같은 폴더의 동시 실행과 손상된 로그는 명시적인 오류로 처리합니다.

`checkpoint.json`은 재개 지점이며, 원시 로그가 최종 근거입니다. 진행 상황은 표준 출력에 기록하고 모델 대기 중에는 턴 하트비트를 남깁니다.

## 결과와 채점

`agentctl benchmark report`는 원시 로그에서 `summary.json`, `summary.ko.md`, `turns.csv`를 생성합니다. 준비 호출을 제외한 그룹별 성공·실패 수, 판단 중앙값, 호출량과 두 정책 모두 성공한 쌍의 비교를 제공합니다. 비용은 확인된 요금 기준이 없으므로 계산하지 않습니다.

답변을 [사전 채점안](rubric.ko.md)과 비교하여 다음 형식의 `grades.json`을 결과 폴더에 저장한 뒤 보고서를 재생성합니다. 채점 주체를 정확히 기록합니다.

```json
[
  {
    "turn_id": "원시 로그의 턴 식별자",
    "reviewer_type": "agent",
    "correct": true,
    "unsupported_visual_claim": false,
    "ambiguity_appropriate": null,
    "notes": "사전 기준과 대조한 근거"
  }
]
```

`reviewer_type=human`은 사람이 실제로 해당 답변을 검토했을 때만 사용합니다. 모델의 `requires_visual` 값은 정답 라벨이 아닙니다. 관측 유무, 장면에 대한 근거 없는 단정, 답변·행동 계획의 정답 여부를 별개로 확인합니다.

이 실험은 작은 고정 입력 파일럿입니다. 실제 브이알챗 화면 전송 비용, 사용자 경험, 아바타 동작 정확도, 추적 주기의 개선을 입증하지 않습니다. 실제 장면과 음성 입력을 연결하는 실험은 별도로 해야 합니다.

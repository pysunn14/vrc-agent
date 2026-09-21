# 교정 측정 월드

추적 입력과 아바타 IK 결과를 같은 프레임에서 비교하는 개발용 로컬 월드입니다. 아바타나 추적 자세를 변경하지 않습니다.

## 준비

VRChat Creator Companion에서 Unity 2022.3.22f1 기반 **Worlds** 프로젝트를 별도로 만듭니다. 이 폴더의 `Runtime`과 `Editor`를 프로젝트의 `Assets/VrcAgentCalibration/` 아래에 복사합니다. Worlds SDK에 포함된 UdonSharp를 사용합니다. 현재 설치 검증 대상 SDK는 3.10.5입니다.

Unity 메뉴 `VRC Agent > Create Calibration Scene`을 실행합니다. 생성되는 장면은 `Assets/VrcAgentCalibration/Scenes/Calibration.unity`입니다. 기존 열린 장면의 저장 여부를 먼저 확인합니다. 생성 장면을 직접 편집했다면 다시 생성할 때 해당 장면이 교체된다는 점에 유의하세요.

SDK의 `Build & Test`로 로컬 VRChat에서 실행합니다. 실제 추적 입력을 측정할 계정이 해당 클라이언트의 로컬 플레이어여야 합니다. 원격 플레이어의 추적 데이터는 같은 의미가 아닙니다. 공개 업로드는 필요하지 않습니다.

## 측정

입장 후 60초 동안 0.5초마다 기록합니다. 녹색 버튼으로 중단하거나 새 측정을 시작합니다. 패널의 샘플 번호와 남은 시간이 동작 상태를 표시합니다.

VRChat 로그에서 `VRC_AGENT_CALIBRATION ` 다음의 JSON을 추출합니다. 종료 표시는 `VRC_AGENT_CALIBRATION_END`, 직렬화 실패는 `VRC_AGENT_CALIBRATION_ERROR`입니다.

- `schema`: 데이터 형식 버전 1.
- `session`, `sequence`, `time_seconds`, `frame`: 측정 구간과 프레임 식별.
- `is_vr`, `eye_height_m`: VR 상태와 현재 아바타 눈높이.
- `player`: 플레이어 위치·회전.
- `tracking`: 머리, 양손, 추적 원점, 아바타 루트.
- `bones`: 머리, 목, 골반, 양쪽 위팔·아래팔·손·발.
- 각 자세는 월드 좌표 `position`과 회전 `quaternion_xyzw`로 표현합니다.

`PostLateUpdate`에서 읽으므로 IK 이후의 골격을 관측합니다. 아바타 로딩이 끝난 뒤 녹색 버튼으로 새 구간을 시작하고, 입력과 출력이 안정된 샘플을 사용하세요. 골격 API의 누락된 뼈 반환값이나 아직 로딩되지 않은 아바타를 유효한 교정 데이터로 간주해서는 안 됩니다. 이 도구 자체는 교정값을 계산하거나 확인된 자세로 등록하지 않습니다.

## 확인할 것

Unity C# 및 Udon 컴파일 성공, 장면 생성 성공, 실제 VRChat 로그의 연속 샘플을 각각 확인해야 합니다. 파일 복사나 장면 생성만으로 게임 내 검증이 완료되지는 않습니다.

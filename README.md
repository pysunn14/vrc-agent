# VRC AGENT

VRChat 에이전트의 제공자 설정, 실행 프로세스와 Windows 연결을 관리하는 CLI입니다. TypeScript·Pi 입력창과 Clack 설정 마법사를 사용하며, 모델 호출과 프로세스 관리는 Python 공통 코어가 담당합니다. 전체 화면 대시보드 없이 일반 터미널 스크롤에 작업과 결과를 남깁니다.

## 설치와 시작

Node.js 22.19 이상, Python 3.11 이상, uv가 필요합니다. 저장소를 내려받은 폴더에서 실행합니다.

```sh
uv sync --locked
npm ci
npm run build
npm link
vrc-agent
```

`npm link`는 현재 개발 폴더를 터미널 명령에 연결합니다. 이후에는 어느 폴더에서나 `vrc-agent`로 실행할 수 있습니다. 개발 폴더는 유지해야 하며, TypeScript 소스를 수정한 뒤에는 `npm run build`로 반영합니다. 다시 연결할 필요는 없습니다. 이 설치 절차는 저장소를 이용하는 방식이며 공개 npm 패키지 설치가 아닙니다.

첫 실행에서 설정 마법사를 엽니다. 다시 설정하려면 `vrc-agent configure`를 사용합니다. 언어는 English·한국어 중 선택하며 `settings ko`, `settings en`으로 변경할 수 있습니다. 일반 터미널 색상을 사용하고 성공·경고·오류를 구분합니다.

에이전트 역할을 선택한 장비에서는 시작 마법사의 **ARDY 준비 방법**에서 **자동 설치 (기본 위치)** 또는 **기존 ARDY 환경 연결**을 선택합니다. 새 설치는 전용 Python 환경과 필요한 모델을 준비하고, 실제 동작 생성 검증까지 수행합니다. VRChat이나 대화 서비스를 자동으로 시작하지 않습니다. 이미 설정을 마쳤다면 `vrc-agent runtime setup`으로 ARDY 단계만 다시 열 수 있습니다. 자세한 절차는 [ARDY 준비](#ardy-준비)를 참고합니다.

핵심 명령은 다음과 같습니다. `vrc-agent` 뒤에 붙이거나, 대화형 입력창에서 `/`로 시작해 입력합니다.

```text
onboard                         초기 설정
configure                       설정 변경
status                          서비스 상태
start provider-<연결ID>          관리 대상으로 등록한 모델 서버 실행
start agent                     대화·음성·ARDY 실행
start bridge                    이 Windows 컴퓨터의 브리지 실행
hosts discover                  Tailscale 장비 검색
connection info                 두 장비의 주소와 공통 포트 확인
stop agent                      러너가 시작한 에이전트 종료
logs agent                      최근 로그
doctor --offline                파일·설정·환경변수 검사
doctor                          실행 환경·서비스·브리지 세션도 검사
providers                       등록한 연결과 기능별 선택
behaviors                       사용 가능한 행동과 준비되지 않은 이유
settings ko                     한국어로 변경
```

Tab으로 명령을 완성하고 위·아래 방향키로 이전 입력을 찾습니다. `/quit`이나 빈 입력창의 Ctrl+D로 콘솔을 닫아도 서비스는 유지됩니다. 종료하려는 서비스를 `stop`으로 지정합니다.

```sh
vrc-agent status --json
vrc-agent --config ./agent.json doctor --offline
vrc-agent --config ./agent.json --state-dir ./state status --json
```

`--python` 또는 `VRC_AGENT_PYTHON`은 CLI가 사용하는 **Python 코어**의 실행 파일입니다. 모델 환경은 설정 파일의 `runtime.python`으로 별도 지정합니다. Python을 실행할 때 가상환경의 실행 경로를 사용해야 하며, 심볼릭 링크의 최종 시스템 Python 경로로 바꾸면 가상환경을 잃을 수 있습니다.

## ARDY 준비

### 새로 설치

마법사에서 **자동 설치 (기본 위치)**를 선택하면 설치 경로를 자동으로 채웁니다. **설치 위치·모델 폴더 지정**을 선택하면 직접 경로를 고를 수 있습니다. 기본값은 운영체제의 사용자 데이터 폴더 아래 `vrc-agent/ardy`입니다. 경로 입력창은 파일·폴더 자동완성을 제공하며, 공백·한글이 들어 있는 폴더와 외장 SSD도 사용할 수 있습니다. 모델만 별도 폴더에 저장할 수도 있습니다.

```text
선택한 설치 폴더/
  source/     고정 버전의 ARDY 소스와 MPS 수정분
  runtime/    전용 Python 가상환경
  models/     버전별 모델과 체크포인트
```

가상환경을 직접 활성화할 필요는 없습니다. 마법사가 Python·모델 경로를 채우며, 모델은 가상환경을 다시 설치해도 재사용할 수 있게 분리합니다. 기존 소스나 가상환경이 있는 폴더는 새 설치 대상으로 사용하지 않습니다.

설치 전에는 다음 항목을 확인합니다.

- `uv`, `git`, C++17 컴파일러가 필요합니다. Mac은 Xcode Command Line Tools, Linux는 C++ 개발 도구, Windows는 Visual Studio 개발자 터미널 등에서 컴파일러를 사용할 수 있어야 합니다. 시스템 개발 도구는 이 마법사가 설치하지 않습니다.
- 새 환경은 Python 3.11.16과 플랫폼별 해시가 포함된 의존성 잠금 파일을 사용합니다. Mac은 Apple Silicon/MPS, Linux·Windows는 x86-64/NVIDIA CUDA 12.8 구성을 제공합니다. Mac에서는 새 설치와 실제 추론을 검증했으며, Linux·Windows의 새 설치 구성은 실제 장비 검증 전으로 표시합니다. 다른 장비 구성은 기존 환경 연결 경로를 사용합니다.
- 기본 모델은 `ARDY-Core-RP-20FPS-Horizon40`입니다. 로컬 문장 인코더의 Llama 및 LLM2Vec 모델까지 포함해 약 16 GiB의 모델 파일을 준비합니다. Python 환경·빌드 파일을 위한 추가 공간도 필요합니다. 기존 Hugging Face 캐시에 같은 버전의 파일이 있으면 무결성을 확인해 재사용합니다.
- 문장 인코더는 대화용 LLM 제공자와 별개입니다. [Llama 모델 페이지](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)의 접근 권한과 Hugging Face 로그인이 필요합니다. 기존 로그인 또는 `HF_TOKEN`을 사용하며, 인증이 필요하면 마법사에서 로그인하고 재개할 수 있습니다. 외부 모델의 접근 승인은 사용자가 해당 페이지에서 완료해야 합니다.

설치는 `소스 → Python 환경 → 모델 → 추론 검증` 순서로 진행합니다. 진행률·작업 상태·로그가 저장됩니다. 검증은 모델 로딩, 문장 인코딩, 연속 두 구간의 동작 생성과 결과 배열의 유효성을 확인합니다. **ARDY 추론 성공은 아바타 자세나 실시간 성능 검증을 의미하지 않습니다.** 검증 과정은 VRChat에 데이터를 전송하지 않습니다.

### 기존 환경 연결

**기존 ARDY 환경 연결**을 선택하고 ARDY 폴더 또는 해당 환경의 Python 실행 파일을 지정합니다. 폴더에서는 `.venv`, `venv`, `runtime` 등의 환경을 찾습니다. 후보가 없거나 여러 개이면 실행 파일을 직접 선택합니다. Python 심볼릭 링크를 시스템 Python으로 치환하지 않습니다.

체크포인트 폴더와 Hugging Face 캐시는 기본적으로 기존 설정을 재사용하며, 고급 설정이나 `runtime setup` 명령의 경로 옵션으로 별도 지정할 수 있습니다. 비워두면 설치된 모델의 기존 캐시를 사용합니다. 기존 환경에서는 패키지 설치·교체나 모델 다운로드를 하지 않고 오프라인 추론을 검증합니다. 검증에 성공해야 해당 경로가 마법사의 실행 설정에 반영됩니다. 이 경로는 로컬 문장 인코더를 사용하므로 필요한 모델이 모두 준비되어 있어야 합니다.

### 중단·재개와 상태 확인

```sh
vrc-agent runtime status
vrc-agent runtime logs
vrc-agent runtime cancel
vrc-agent runtime resume --wait
vrc-agent runtime auth
```

터미널을 닫거나 설치 대기 화면에서 Ctrl+C를 누르면 백그라운드 작업은 유지됩니다. 설치 자체를 중단하려면 `runtime cancel`을 사용합니다. 다시 `runtime setup`을 열고 **이전 설치 확인·재개**를 선택하면 성공 결과를 설정에 연결할 수 있습니다. 인증 대기 상태에서는 `runtime auth`로 로그인한 뒤 `runtime resume`으로 재개합니다.

기록된 단계만 믿고 건너뛰지 않습니다. 소스·환경·모델의 실제 상태를 다시 확인하며, 깨진 파일을 준비 완료로 처리하지 않습니다. 프로세스가 사라지면 중단 상태로 표시하고, 이전 자식 프로세스가 남았다면 취소 명령으로 정리한 뒤 재개하도록 안내합니다. 설치 오류와 마지막 단계는 `runtime status`, 상세 출력은 `runtime logs`에 남습니다.

자동화에서는 폴더를 명시해 작업을 시작합니다. 상대 경로는 명령을 실행한 폴더 기준으로 저장됩니다.

```sh
vrc-agent runtime plan --root "./ardy installation" --json
vrc-agent runtime setup --root "./ardy installation" --wait
vrc-agent runtime setup --existing "./existing ardy" --wait
vrc-agent runtime verify --wait
```

`--root` 등의 옵션을 사용하는 설치 명령은 작업 결과를 보관하며 활성 프로필을 자동으로 교체하지 않습니다. 마법사의 **이전 설치 확인·재개**에서 결과를 선택해 최종 저장합니다. `runtime verify`는 현재 프로필의 환경을 다시 검사합니다. `--python`은 CLI 코어용, `--runtime-python`은 연결할 ARDY 환경용입니다. 상태·로그·재개·취소 명령에는 작업 ID도 지정할 수 있습니다.

설치 명세는 `vrc_ardy_agent/runner/runtime_setup/recipes/`에 있습니다. 소스 커밋과 MPS 패치, 플랫폼별 의존성, 모델 저장소의 고정 리비전·파일 해시를 함께 관리합니다. 새 설치는 관리 대상 환경의 잠금 파일을 적용하며, 평소 에이전트 실행은 준비된 로컬 모델을 사용합니다. `doctor`는 가벼운 환경 진단을 유지하고 실제 추론은 `runtime verify`로 요청합니다.

## 설정 구조

실행 설정은 `version: 3` 형식이며 운영체제의 사용자 설정 폴더에 `vrc-agent/agent.json`을 저장합니다. `--config`로 다른 파일을 선택할 수 있습니다.

| 구분 | 역할 |
| --- | --- |
| `role` | 이 컴퓨터의 실행 역할: `agent`, `bridge`, `combined` |
| `connection` | 양쪽에서 맞춰 사용하는 `stream_port`·`agent_port`·`bridge_port` |
| `hosts` | 컴퓨터의 운영체제와 접속 주소 |
| `runner_host`, `game_host` | 모델 실행기와 Windows 게임의 위치 |
| `providers` | 제공자 종류, 연결 이름, 주소·인증 설정, 실행 관리 방식 |
| `bindings` | 대화·전사·음성 합성에서 사용할 연결 ID와 모델·목소리 |
| `runtime` | 에이전트 역할에서 사용하는 ARDY 환경과 실행 장치 |
| `avatar` | 골격·기준 위치·기준 자세·표정 채널과 교정 기록 |
| `behaviors` | 저장 동작·ARDY 생성·이동 행동의 정의 |
| `autonomy` | 연속 대기 시간과 자동으로 실행할 행동 목록 |
| `bridge` | 로컬 Windows 브리지의 실행 환경, 계정·오디오 장치·몸 추적 출력 |

`agent`·`combined`에는 `providers`, `bindings`, `runtime`, `avatar`, `behaviors`, `autonomy`가 있습니다. `bridge`·`combined`에는 로컬 `bridge` 설정이 있습니다. 연결 전용 역할은 `bridge.avatar_rig`와 `bridge.hmd_base`로 Windows 쪽 골격·기준 위치를 지정합니다. 통합 역할은 `avatar`의 값을 함께 사용합니다. 원격 장비의 로컬 Python·파일 경로는 이 장비의 설정에 요구하지 않습니다.

같은 제공자 종류를 서로 다른 주소·계정으로 여러 번 등록할 수 있습니다. 모델과 목소리는 연결에 종속된 기능별 설정입니다. 인증이 없으면 마법사에서 **API 키를 가려서 입력**합니다. 키는 운영체제 자격 증명 저장소에 저장하고 설정·초안에는 참조만 남깁니다. macOS는 Keychain, Windows는 Credential Locker, Linux는 Secret Service를 사용합니다. 저장소가 없는 서버 환경에서는 기존 환경변수 방식을 선택할 수 있으며, 평문 파일로 자동 전환하지 않습니다.

로컬 Hermes에 연결하며 `API_SERVER_KEY`가 현재 환경에 없다면 `HERMES_HOME` 또는 기본 Hermes 폴더의 `.env`를 찾아 참조합니다. 파일의 키를 복사할 필요가 없습니다. 원격 Hermes 주소에는 이 로컬 인증을 자동 사용하지 않습니다. 기존 인증이 거절되면 다른 인증으로 몰래 바꾸지 않고 키 등록·주소 수정 선택을 보여줍니다.

마법사·모델 목록 조회·진단·실제 실행은 같은 인증 해석을 사용합니다. ARDY의 분리된 Python 환경에는 실행 직전에 연결별 인증을 메모리상의 환경변수로 전달하므로 그 환경에 별도 키 저장소 패키지를 설치할 필요가 없습니다. 새 키 등록은 새 참조를 만들며, 초안을 취소해도 기존 활성 설정이 참조하는 키는 바뀌지 않습니다.

마법사는 첫 실행에서 언어를 묻고, 기존 설정이 있으면 요약과 항목별 수정 화면으로 바로 들어갑니다. 답변과 완료한 단계를 초안에 저장하므로 재개할 때 완료한 단계를 다시 입력하지 않습니다. 포트·제한 시간·실행 관리 방식·교정 좌표는 요약 화면의 **고급 설정**에서 바꿉니다. 취소한 초안은 다음 설정에서 다시 불러올 수 있으며, 최종 검증과 저장 전에는 실행 설정이 바뀌지 않습니다. 다른 클라이언트가 저장한 설정을 오래된 초안으로 덮어쓸 수 없습니다. 관리 중인 프로세스가 있으면 실행 구성을 바꾸기 전에 해당 프로세스를 종료해야 합니다. 언어 변경은 실행에 영향을 주지 않습니다.

`profile export`로 현재 설정을 출력하고, `profile import FILE`로 파일을 검증한 뒤 저장할 수 있습니다. 고급 연결 배치와 포트는 이 경로에서도 설정할 수 있습니다. 상대 동작 파일·작업 폴더는 설정 파일의 폴더를 기준으로 해석합니다. 프로세스 명령은 셸 문자열이 아닌 **JSON 인자 배열**입니다.

## 제공자

| 제공자 ID | 지원 기능 | 연결 방식 |
| --- | --- | --- |
| `openai` | 대화·전사·음성 합성 | 제공자가 지원하는 모델의 Chat Completions·Audio API |
| `openai-compatible` | 대화·전사·음성 합성 | 선택한 서버가 구현한 호환 API 기능 |
| `gemini` | 대화 | Gemini의 OpenAI 호환 API |
| `hermes` | 대화 | 상위 제공자와 추론 설정을 전달하는 Hermes 어댑터 |
| `crisperwhisper` | 전사 | 호환 전사 API와 전용 전사 모드 |
| `gpt-sovits` | 음성 합성 | `/audio/speech`를 제공하는 호환 서버 |
| `elevenlabs` | 음성 합성 | 전용 API, 24 kHz PCM을 WAV로 변환 |
| `whisper-local` | 전사 | 모델 실행 환경 내부의 faster-whisper |

서비스 이름은 **설정을 돕는 프리셋**이며, 연결 구현과 모델 등록 여부는 별개입니다. OpenAI·Gemini·Hermes·CrisperWhisper·GPT-SoVITS는 공통 호환 구현을 사용하고 필요한 요청 옵션만 확장합니다. ElevenLabs와 로컬 Whisper는 별도 규격으로 처리합니다. 저장한 연결은 여러 기능에서 재사용할 수 있으며, 프리셋의 기본값이 이미 저장한 모델·목소리·옵션을 덮어쓰지 않습니다.

마법사는 저장된 연결을 우선 제안합니다. 새 연결은 **서비스 선택 / 서버 주소로 연결 / 이 컴퓨터에서 직접 실행**으로 나눕니다. 직접 실행은 실제 지원하는 로컬 엔진이 있을 때만 표시합니다. 모델은 서버에서 조회해 하나면 자동 선택하고, 여러 개면 검색해서 고릅니다. 모델을 선택했다는 사실과 해당 기능의 실제 시험 성공은 구분합니다.

목소리 목록은 서버가 명시한 경로로 조회합니다. 일반 호환 서버는 고급 설정의 `voices_path`를 지정하거나 음성 ID를 입력합니다. 상대 경로는 API 주소 아래에 붙고, `/`로 시작하는 경로는 같은 서버의 루트 기준입니다. 프록시 경로도 직접 지정할 수 있으며 다른 호스트로 인증을 보내는 주소는 허용하지 않습니다. GPT-SoVITS 프리셋은 이 프로젝트가 사용한 호환 변환 서버의 `/health` 목록과 길이 미정 스트리밍 WAV 응답 계약을 사용합니다. 다른 GPT-SoVITS 배포는 실제 서버 계약에 맞춰 일반 호환 연결을 사용하세요. OpenAI의 내장 목소리는 문서 기반 프리셋 선택지로 표시하고 실제 모델과의 호환성은 시험으로 확인합니다.

인증 누락·인증 거절·연결 실패·빈 목록·목록 API 미지원은 원인을 구분합니다. 실패해도 저장된 모델을 지우거나 다른 모델로 바꾸지 않습니다. 사용자 지정 서버의 `request_extension`을 고급 설정에서 선택하면 Hermes 요청 형식이나 Crisper 전사 옵션을 명시적으로 사용할 수 있습니다.

요약에서 **연결 시험 후 저장**을 선택하면 실제 API 요청으로 대화·음성 인식·음성 합성을 확인합니다. ARDY·VRChat을 실행하지 않으며 마이크를 녹음하거나 생성 음성을 재생하지 않습니다. 화면 사용을 선택한 LLM은 합성 이미지 요청도 확인합니다. 기본 STT 시험에는 출처와 라이선스가 포함된 [영어 음성 샘플](vrc_ardy_agent/providers/assets/README.md)을 사용합니다. 다른 언어를 지정한 경우 해당 언어의 16 kHz 모노 WAV를 선택합니다. 시험은 요청과 응답의 규격 확인이며 모델의 인식 품질을 보장하지 않습니다. 시험 없이 설정을 저장할 수도 있습니다.

```bash
vrc-agent providers check llm
vrc-agent providers check stt --audio ./sample-16khz-mono.wav
vrc-agent providers check tts --text "안녕하세요"
vrc-agent providers observations llm
```

`providers check`는 HTTP 연결만 시험하므로 ARDY 환경을 요구하지 않습니다. 로컬 Whisper는 지정한 실행 환경에서 시험합니다. `providers test`는 기존 실행 환경에서 에이전트 응답 계약까지 확인하는 별도 시험입니다. 마지막 시험은 설정과 분리해 기록하며 주소·인증 참조·모델·옵션이 바뀌면 다른 시험 대상으로 취급합니다. **과거 성공 기록은 현재 서버 상태를 보장하지 않습니다.** 환경변수나 외부 인증 파일의 내용 변경도 새 조회·시험으로 확인해야 합니다.

`providers models <연결ID>`와 `providers voices <연결ID>`도 마법사와 같은 조회 기능을 사용합니다. 다음 페이지가 있으면 같은 명령에 `--cursor <토큰>`을 붙입니다. 공식 API 기준은 [OpenAI 모델 목록](https://developers.openai.com/api/reference/resources/models/methods/list), [ElevenLabs 모델 목록](https://elevenlabs.io/docs/api-reference/models/list), [ElevenLabs 목소리 목록](https://elevenlabs.io/docs/api-reference/voices/search)입니다. Hermes에서 조회한 모델 별칭은 서버에 설정된 경로를 사용하며, 상위 제공자를 강제로 바꾸는 옵션은 고급 설정에 있습니다.

새 API는 제공자 정의와 어댑터를 추가해 지원합니다. 포트 번호가 같다는 이유로 서로 다른 규격의 API를 호환 제공자로 취급하지 않습니다. 로컬 Qwen 등의 서버도 해당 기능의 호환 규격을 실제로 구현한 경우에 연결할 수 있습니다.

실행 관리 방식은 다음 네 가지입니다.

- `api`: 외부 API를 호출합니다. 프로세스의 시작·종료를 소유하지 않습니다.
- `external`: 지정한 장비에서 이미 실행 중인 서비스에 연결합니다.
- `managed`: 현재 러너 장비에서 포그라운드 명령을 실행하고 프로세스를 관리합니다.
- `embedded`: 모델을 에이전트 프로세스 내부에서 로드합니다. 현재 로컬 Whisper가 사용합니다.

Windows 음성 설정은 선택한 Python 환경에서 입력·출력 장치를 검색합니다. 가상 마이크로 보낼 출력 장치는 사용자가 목록에서 고릅니다. 이름이 중복되면 번호를 저장하고, 장치 구성 변경 뒤 다시 확인해야 한다고 안내합니다. 이 검색은 음성을 재생하거나 VRChat을 시작하지 않습니다.

로컬 Whisper는 실행 환경에 `local-stt` 선택 의존성과 모델이 준비되어 있어야 합니다. CPU·CUDA를 선택할 수 있으며, MPS는 이 어댑터의 지원 장치가 아닙니다. 모델이 없을 때 임의로 다운로드하지 않습니다. 모델 ID 또는 실행 작업 폴더 기준의 설치된 모델 경로를 사용합니다.

## 장비 배치와 Windows 브리지

현재 운영체제는 자동으로 감지합니다. Mac·Linux에서는 이 장비가 에이전트를 실행한다는 안내 후 Windows 장비를 선택합니다. Windows에서는 다음 역할을 선택합니다.

| 이 컴퓨터의 역할 | 이 컴퓨터에서 설정하는 항목 | 로컬 실행 대상 |
| --- | --- | --- |
| 에이전트와 VRChat 함께 실행 | 제공자·ARDY·아바타·행동·Windows 브리지 | `agent`, `bridge`, 관리 제공자 |
| 에이전트만 실행 | 제공자·ARDY·아바타·행동·Windows 접속 주소 | `agent`, 관리 제공자 |
| VRChat 연결만 담당 | 에이전트 접속 주소·브리지 Python·골격·음성·추적 | `bridge` |

대화·전사·음성 합성 제공자는 각각 교체할 수 있습니다. ARDY는 에이전트 장비에서 실행하며, 다른 장비의 모델 서버나 외부 API는 제공자 설정으로 연결합니다. Windows 연결 전용 역할에는 ARDY나 대화 제공자 설정이 없습니다.

### 장비 선택과 연결

같은 Windows 컴퓨터에서 모두 실행하면 루프백 주소를 사용하며 Tailscale 검색과 주소 질문을 건너뜁니다. 별도 장비 구성에서는 로컬 Tailscale의 장비 목록을 읽습니다. 장비 이름·주소·온라인 여부를 보여주고, 선택하면 상대 주소와 현재 장비의 주소를 함께 채웁니다. 꺼진 장비도 선택할 수 있지만 **목록의 온라인 표시는 VRChat이나 브리지 실행 성공을 뜻하지 않습니다.**

Tailscale이 없거나 로그인되지 않았거나 검색에 실패하면 이유를 표시합니다. 다시 검색하거나 내부망·Tailscale 주소를 직접 입력할 수 있습니다. 직접 입력에서는 양쪽 주소를 받으며, 연결 전용 Windows에서 상대 에이전트의 운영체제를 모르면 사용자가 지정합니다. 현재 컴퓨터의 운영체제를 묻는 질문은 없습니다. 직접 입력한 내부망 주소에 Tailscale 주소를 섞어 넣지 않습니다.

검색은 `tailscale status --json`만 실행합니다. macOS 앱에 포함된 CLI, Windows 설치 폴더 또는 PATH의 CLI를 찾으며 Tailscale 설치·로그인·DNS·Serve 설정을 변경하지 않습니다.

```sh
vrc-agent hosts discover
vrc-agent connection info
```

양쪽 장비에서 `onboard`를 실행합니다. 기본 연결 포트는 동작·음성 스트림 `8766`, 에이전트 상태 `8765`, Windows 상태 `8767`입니다. 마법사에서 변경할 수 있으며 두 장비에서 같은 값을 사용해야 합니다. 상태 서버는 설정한 해당 장비 주소로 연결을 받습니다. 이 주소는 상대 장비에서 접속할 수 있어야 합니다.

### Windows 브리지 준비와 실행

Windows에도 동일한 저장소와 CLI를 설치하고 `vrc-agent onboard`에서 역할을 선택합니다. 마법사는 선택한 로컬 프로젝트의 `.venv`, `venv`, `runtime`에서 Python을 찾습니다. Windows에서는 `Scripts/python.exe`, Mac·Linux에서는 `bin/python`을 사용합니다. 후보가 없거나 여러 개이면 이유를 표시하고 다른 환경을 선택받습니다. 프로젝트와 Python 경로가 실제로 존재하는지 확인하며, 현재 CLI의 Python을 선택한 경우에도 브리지 의존성 검사는 별도로 수행합니다.

브리지 Python에는 `requirements-windows.txt`의 실행 의존성과 `Pillow`가 필요합니다. VRChat·SteamVR·VMT·가상 오디오 장치도 Windows에서 준비해야 합니다. 프로세스 오디오 입력을 사용하면 [Windows 오디오 도우미](native/windows_process_audio_probe/src/VrcArdy.ProcessAudioProbe.csproj)를 빌드합니다. 이 마법사는 게임·드라이버·Windows 의존성을 자동으로 설치하지 않습니다. ARDY 환경과 브리지 환경은 별도로 지정할 수 있습니다.

여러 VRChat 창을 사용할 때는 대상 계정 이름이나 `bridge.vrchat_pid`를 명시합니다. 몸 추적 출력은 VRChat OSC 또는 VMT를 선택합니다. 별도 Windows 구성에서는 에이전트와 같은 아바타 설정 묶음을 가져오면 골격과 HMD 기준 위치를 함께 적용합니다. 통합 역할에서는 골격·기준 위치를 중복 입력하지 않습니다.

Windows에서 실행합니다.

```sh
vrc-agent doctor --offline
vrc-agent doctor
vrc-agent start bridge
vrc-agent status
vrc-agent stop bridge
```

에이전트 장비에서는 필요한 관리 제공자와 `agent`를 시작합니다. `status`는 양쪽 서비스를 관측하며, `start`·`stop`은 현재 역할이 소유하는 로컬 서비스만 조작합니다. 원격 SSH 설치·실행은 포함하지 않습니다. 콘솔에서 서비스 이름 없이 `start`·`stop`을 입력하면 로컬 관리 대상만 선택지로 나옵니다.

`vrc-agent bridge command`는 **Windows 브리지 역할의 장비에서** PowerShell 실행 명령을 출력합니다. 에이전트 전용 장비에서는 `connection info`로 연결 정보를 확인합니다. 러너와 Windows 브리지는 같은 버전의 소스를 사용해야 합니다. 스트림 연결 규격은 `version: 2`이며 프로필 버전과 별개입니다. 다른 규격의 브리지는 자세 출력 전에 연결 단계에서 거부합니다.

## 아바타와 행동 설정

마법사에서 준비된 아바타 설정 묶음을 선택하거나 가져옵니다. 새 아바타는 [Unity에서 내보낸 골격 JSON](native/unity_avatar_rig_exporter/README.md)을 가져오면 기준 자세 후보가 생성됩니다. `vrc-agent avatars calibrate`에서 Windows로 후보를 전송하고 손 위치·회전을 조정할 수 있습니다. **추적이 켜진 상태에서 자세를 직접 확인한 뒤** 저장·등록합니다. Windows가 준비되지 않았다면 초안을 보관하고 나중에 이어갈 수 있습니다.

교정 기록에는 골격·기준 자세·기준 위치·표정 연결의 지문을 저장합니다. 이 내용이 바뀌면 `doctor`와 실행 전 검사에서 다시 확인하도록 표시합니다. 이 검사는 파일과 사람의 확인 기록이 일치하는지 검사하며, 현재 게임 화면을 자동으로 판정하지 않습니다.

`avatar.face_channels`는 행동이 사용하는 채널 이름을 아바타의 실제 OSC 파라미터에 연결합니다. 예를 들어 `{"expression": "Expression"}`으로 설정하면 행동의 `expression` 트랙을 `/avatar/parameters/Expression`으로 보냅니다. 해당 파라미터를 아바타의 표정 애니메이션에 연결하는 Unity 설정은 별도로 필요합니다. 표정을 쓰지 않으면 `{}`로 둡니다.

등록 행동은 선택 사항입니다. 기준 자세만 준비해도 실행할 수 있으며, 하품 같은 특정 행동 파일을 필수로 요구하지 않습니다. [행동 목록 예제](examples/behaviors.json)를 마법사의 **행동 목록 JSON 가져오기**에서 선택할 수 있습니다. 파일의 상대 경로는 행동 목록 파일의 폴더를 기준으로 해석하고, 가져온 정의를 실행 설정에 저장합니다.

| 실행 방식 | 정의 | 완료 기준 |
| --- | --- | --- |
| `clip` | `frames`, 선택적인 `face_tracks` | 저장 프레임을 정확히 한 번 출력 |
| `ardy` | `prompt`, `duration_seconds` | 실제 재생 시작 후 지정 시간 |
| `locomotion` | `velocity`, `duration_seconds` | 실제 이동 출력 시작 후 지정 시간 |

`velocity`는 `[좌우, 앞뒤, 회전]` 입력값이며 각 값은 -1부터 1까지입니다. 현재 검증한 짧은 이동은 앞뒤에 ±0.18을 사용합니다. 이동 중 팔 애니메이션을 VRChat에 맡기는 처리는 계속 적용됩니다. 이는 목표 지점 탐색이나 자율 경로 계획 기능이 아닙니다.

저장 동작을 추가하는 행동 목록 예시는 다음과 같습니다. 아바타 파일은 저장소에 포함하지 않습니다.

```json
{
  "version": 1,
  "behaviors": {
    "yawn": {
      "source": "clip",
      "description": "하품을 한 번 합니다.",
      "frames": "./motions/yawn.frames.json",
      "face_tracks": {"expression": "./motions/yawn.face.json"}
    }
  }
}
```

프레임 파일은 `six_point_pose` 객체의 배열입니다. 표정 파일은 프레임 수와 길이가 같은 0부터 1까지의 숫자 배열이며, 여러 채널은 각각 별도 트랙을 사용합니다. 프레임의 FPS·크기 배율·활성 추적기는 기준 자세와 같아야 합니다. 저장 동작에 월드 이동 입력을 섞을 수는 없습니다. 시작과 끝이 기준 자세와 달라도 실행기가 전환을 보간합니다.

저장 동작을 가져올 때는 해당 아바타에서 추적을 켠 채 확인한 동작인지 따로 확인합니다. 확인한 동작에는 아바타 지문과 동작·표정 파일 지문을 기록합니다. 이후 파일이 바뀌거나 채널 연결이 없으면 해당 행동을 사용할 수 없는 이유를 표시합니다. 다른 행동은 계속 사용할 수 있으며, 요청된 행동을 임의로 다른 동작이나 ARDY 생성으로 대체하지 않습니다.

```sh
vrc-agent behaviors
vrc-agent doctor --offline
```

모델에 알려주는 행동 목록, 제어 API의 `/actions`, 실행 검증은 같은 준비된 목록을 사용합니다. 등록 행동은 `{"type":"motion","name":"yawn"}`처럼 요청합니다. 저장 동작은 시간을 지정하지 않고 한 번 재생하며, 생성·이동 행동은 정의된 기본 시간 또는 1부터 10초까지의 요청 시간을 사용합니다.

공통 실행기는 하나의 자세 출력 경로를 유지합니다. 동작 전환과 정상 완료 때 위치를 보간하고 회전은 구면 선형 보간으로 연결합니다. 정상 완료 후에는 검증한 기준 자세로 돌아옵니다. 이동 종료 시 이동 입력은 즉시 0으로 만들고, 사용한 표정 파라미터도 중립으로 복귀합니다. 명시적 전체 정지와 연결 끊김·출력 실패는 정상 완료와 구분하여 제어권을 회수합니다.

`autonomy`는 기본적으로 꺼져 있습니다. 마법사에서는 행동 하나와 연속 대기 시간(기본 30초)을 선택할 수 있습니다. 여러 행동을 번갈아 실행하려면 설정 파일에 목록을 지정합니다.

```json
{"enabled": true, "interval_seconds": 30, "behaviors": ["yawn", "wave"]}
```

음성 입력·전사·대화 처리·발화·몸 동작이 진행 중이거나 출력 연결이 준비되지 않았으면 자율 행동을 시작하지 않습니다. 실제로 조용한 상태가 이어진 시간을 세고, 사용 가능한 행동만 순서대로 선택합니다. 실패 상태에서 자동 재시도를 반복하지 않으며, 상태 API의 `components.autonomy`와 동작 상태에서 원인을 확인할 수 있습니다. `--manual-actions-only`로 실행하면 자율 행동도 꺼집니다.

## 진단과 재현 정보

`doctor --offline`은 서비스를 호출하거나 모델을 로드하지 않습니다. 일반 `doctor`는 역할별로 검사합니다. 에이전트에서는 Python 패키지·GPU·ARDY를, 연결 전용 Windows에서는 브리지 Python·음성 장치·오디오 도우미를 확인합니다. 양쪽 서비스 상태와 실제 브리지 세션도 검사합니다. 추적기 등록처럼 자동으로 확정할 수 없는 상태는 미검증으로 남깁니다. 포트가 열려 있다는 이유만으로 추론 성공이나 게임 동작 성공을 표시하지 않습니다.

실제 제공자 호출은 샘플을 명시해서 검사합니다. 외부 API를 선택했다면 일반 API 호출처럼 과금될 수 있습니다.

```sh
vrc-agent providers test llm --text "Hello"
vrc-agent providers test stt --audio ./sample-16khz-mono.wav
vrc-agent providers test tts --text "Hello"
vrc-agent runtime probe
vrc-agent snapshot
```

제공자 검사는 선택한 모델 실행 환경에서 수행합니다. 대화 결과의 동작을 실행하거나, TTS 결과를 재생하거나, VRC에 데이터를 전송하지 않습니다. 전사 입력은 16 kHz·16비트 PCM·모노 WAV입니다. 실패하면 원인을 반환하며 다른 제공자로 조용히 바꾸지 않습니다.

연결 전용 역할의 `snapshot`은 Windows 브리지 환경과 로컬 골격을 기록합니다. 에이전트 역할의 `snapshot`은 프로필 식별값, 실행 환경의 버전, 확인 가능한 ARDY 소스 버전, 선택한 모델·목소리, 동작 파일과 의존성 잠금 파일의 해시를 `agent.lock.json`에 기록합니다. 진행 중에는 `.pending` 체크포인트를 남기며 실패 시 이전 완료 기록을 보존합니다. 클라우드 모델 ID와 외부 서비스는 제공자 쪽에서 바뀔 수 있으며, 이 기록이 외부 서비스나 모델 가중치 자체를 보관하지는 않습니다.

JSON 상태 값은 표시 언어와 무관합니다. doctor 오류는 종료 코드 1, 설정·조작 오류는 2, 마법사 취소는 130입니다. `running`은 프로세스 생존, `ready`는 설정한 상태 주소의 성공, `external`은 러너 외부에서 실행된 서비스를 뜻합니다. 브리지의 연결 여부와 모델 추론 검증은 별도 검사입니다.

## 개발과 검증

```sh
npm run check
npm test
uv run --group runtime-test pytest -q
npm run agentctl -- actions
```

`cli/dev/agentctl.ts`는 화면과 같은 코어를 호출하는 개발용 도구입니다. 상태·로그·설정·기능 조회와 여러 동작의 동시 예약을 제공합니다. 예약은 `--input`의 JSON 배열을 읽고 `--output`에 진행 체크포인트와 결과를 남깁니다. 개발용 파일은 CLI 배포 빌드에 포함하지 않습니다.

```json
[
  {"at_seconds": 0, "action": "start", "payload": {"service": "provider-local"}},
  {"at_seconds": 0, "action": "start", "payload": {"service": "provider-local"}},
  {"at_seconds": 2, "action": "status"}
]
```

Python 단위·통합 테스트, npm 코어 연결과 마법사 테스트, Mac의 실제 터미널 입력, 격리된 프로세스 동시 실행·종료와 로컬 HTTP 제공자 경로를 검증합니다. Linux·Windows의 실제 모델·VRChat 구동과 각 외부 제공자의 실제 계정 호출은 별도 장비·서비스 검증이 필요합니다.

제공자 프리셋은 `vrc_ardy_agent/providers/presets.py`, 통신 계약은 `providers/protocols.py`, 요청 확장은 `providers/extensions.py`에 있습니다. `providers/catalog.py`가 이를 조합하며, 호출 구현은 `providers/adapters.py`, 구성·진단·실행은 `runner/`, 터미널은 `cli/src/`에 있습니다. 프런트엔드가 Python 코어에서 정의를 읽으므로 설정 화면과 실제 실행에 서로 다른 제공자 목록을 유지하지 않습니다.

참고한 공개 인터페이스: [Pi](https://github.com/earendil-works/pi/tree/main/packages/tui), [Clack](https://bomb.sh/docs/clack/packages/prompts/), [Gemini 호환 API](https://ai.google.dev/gemini-api/docs/openai), [OpenAI 음성 API](https://developers.openai.com/api/docs/guides/text-to-speech), [ElevenLabs 음성 API](https://elevenlabs.io/docs/api-reference/text-to-speech/convert), [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## 아바타 설정 등록

준비된 아바타는 골격·기준 자세·표정 연결을 묶어 등록합니다.
`vrc-agent configure`에서 가져오거나 현재 검증한 설정을 등록할 수 있습니다.
목록은 `vrc-agent avatars list`로 확인합니다.
제작·등록·공유 절차와 교정 기능의 범위는 [아바타 설정 안내](docs/avatar-settings.md)를 참고하세요.

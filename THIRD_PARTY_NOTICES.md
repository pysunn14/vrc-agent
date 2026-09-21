# 외부 구성요소

이 저장소의 자체 코드는 AGPL-3.0-only로 배포합니다. 외부 구성요소의 권리·라이선스는 그대로 유지됩니다. 아래는 주요 구성요소이며 각 의존성에 포함된 원문 고지도 함께 확인해야 합니다.

## ARDY 소스와 수정 패치

- 출처: https://github.com/nv-tlabs/ardy
- 설치 기준 리비전: `693f74d13b3d04a0a22ce127ee79c929dd89756b`
- 저작권: NVIDIA CORPORATION & AFFILIATES.
- 원본 라이선스: Apache-2.0. [원문](LICENSES/Apache-2.0.txt).
- `vrc_ardy_agent/runner/runtime_setup/recipes/mps.patch`는 장치 선택과 Apple Silicon 실행 지원을 위해 작성한 수정 패치입니다. 원본 코드의 저작권·라이선스 표시는 유지합니다. 패치의 추가 변경은 VRC Agent Contributors가 작성했습니다.

ARDY 모델 가중치는 이 저장소에 포함하지 않습니다. 모델의 이용 조건은 소스 코드와 별개입니다. 설치 시 내려받는 모델의 모델 카드와 라이선스를 확인해야 합니다. ARDY 상위 저장소는 모델에 NVIDIA Open Model License를 안내하며, Llama 텍스트 인코더에는 별도 접근 승인과 해당 모델 이용 조건이 적용됩니다.

## YOLO 추적

`requirements-windows.txt`의 Ultralytics와 해당 YOLO 모델은 상위 제공자의 AGPL-3.0 또는 별도 Enterprise 조건을 따릅니다. 모델 가중치는 이 저장소에 포함하지 않습니다.

- 소스: https://github.com/ultralytics/ultralytics
- 라이선스 안내: https://www.ultralytics.com/license

## 전사 연결 시험용 음성

`speech-sample.wav`는 LibriSpeech 자료를 변환한 음성으로 CC BY 4.0이 적용됩니다. 출처, 저자, 변환 내용과 해시는 [샘플 안내](vrc_ardy_agent/providers/assets/README.md)와 인접 manifest에 있습니다. 프로젝트의 AGPL 라이선스로 재지정하지 않습니다.

## CLI 의존성

Clack, Pi TUI, Chalk, Figlet은 각 배포본의 MIT 라이선스와 고지를 유지합니다. 정확한 설치 버전은 `package-lock.json`에 기록합니다. 이 목록이 잠금 파일에 포함된 모든 전이 의존성의 고지를 대체하지는 않습니다.

## 사용자 준비 자료

Unity·VRChat SDK와 구매 아바타, 사용자 음성·골격·교정 파일은 사용자가 각각의 권한과 이용 조건에 따라 준비합니다. 아바타 데이터는 이 저장소의 배포 대상이 아닙니다. 외부 서비스의 계정·API 키·유료 이용 권한도 제공하지 않습니다.

# Unity avatar rig profile exporter

VRChat Humanoid 아바타의 현재 편집기 자세를 런타임 리타게팅용 JSON으로 추출한다.

## 설치

`Editor/AvatarRigProfileExporter.cs`를 Unity 프로젝트의
`Assets/VrcArdy/Editor/AvatarRigProfileExporter.cs`로 복사한다.

## 사용

1. Unity가 Play Mode가 아닌지 확인한다.
2. 애니메이션 미리보기를 끄고 아바타를 기준 자세로 돌린다.
3. Hierarchy에서 Humanoid `Animator`가 있는 아바타나 그 자식을 선택한다.
4. `Tools > VRC Ardy > Export Selected Avatar Rig Profile`을 누른다.

결과는 `Assets/VrcArdy/Generated/<avatar>.avatar-rig.json`에 생성된다.

좌표는 아바타 루트 축을 따르며 머리가 원점이다. 길이에는 Unity 장면에 실제로
적용된 스케일이 반영된다. 골격 회전은 측정 자료이며 VMT에 바로 보내는 명령이
아니다. 실제 추적기 장착 방향 보정은 런타임 어댑터가 담당한다.

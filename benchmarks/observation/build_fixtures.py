"""Create owned, deterministic diagram stimuli, not recordings of VRChat users."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent

def main():
    image_dir = ROOT / 'images'
    image_dir.mkdir(exist_ok=True)
    drawings = {
        'mixed': [('box', (100, 140, 260, 300), '#df3030'), ('ball', (400, 140, 560, 300), '#2059dc')],
        'three': [('box', (60, 140, 200, 300), '#1e9c48'), ('box', (250, 140, 390, 300), '#1e9c48'), ('box', (440, 140, 580, 300), '#1e9c48')],
        'two_balls': [('ball', (100, 140, 260, 300), '#df3030'), ('ball', (400, 140, 560, 300), '#df3030')],
        'empty': [],
    }
    hashes = {}
    for name, shapes in drawings.items():
        image = Image.new('RGB', (640, 480), '#eeeeee')
        draw = ImageDraw.Draw(image)
        for shape, bounds, color in shapes:
            if shape == 'box': draw.rectangle(bounds, fill=color, outline='#222222', width=3)
            else: draw.ellipse(bounds, fill=color, outline='#222222', width=3)
        path = image_dir / f'{name}.jpg'
        image.save(path, quality=95, subsampling=0, optimize=False)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    specs = [
        ('n01','nonvisual','안녕, 루나!','mixed','none','짧게 인사한다. 장면 외형을 덧붙이지 않는다.'),
        ('n02','nonvisual','고마워, 덕분에 도움이 됐어.','mixed','none','감사에 자연스럽게 답한다.'),
        ('n03','nonvisual','오늘 공부 시작하려는데 한마디 응원해줘.','mixed','none','공부를 응원한다. 현재 표정이나 외형을 단정하지 않는다.'),
        ('n04','nonvisual','2 더하기 3은 얼마야?','mixed','none','5라고 답한다.'),
        ('n05','nonvisual','사과를 영어로 뭐라고 해?','mixed','none','apple이라고 답한다.'),
        ('n06','nonvisual','오른손으로 짧게 손 흔들어줘.','mixed','none','오른손을 흔드는 ardy_motion을 계획한다.'),
        ('n07','nonvisual','가볍게 고개를 끄덕여줘.','mixed','none','고개를 끄덕이는 ardy_motion을 계획한다.'),
        ('n08','nonvisual','하품 한 번 해줘.','mixed','none','등록된 motion/yawn을 정확히 한 번 계획한다.'),
        ('n09','nonvisual','제자리에서 짧게 박수 쳐줘.','mixed','none','박수 ardy_motion을 계획하며 이동을 추가하지 않는다.'),
        ('n10','nonvisual','앞으로 3초 걸어줘.','mixed','none','motion/walk_forward를 duration_seconds=3으로 계획한다.'),
        ('v01','visual','화면 왼쪽 사각형은 무슨 색이야?','mixed','required','왼쪽 사각형이 빨간색이라고 답한다.'),
        ('v02','visual','화면 오른쪽 원은 무슨 색이야?','mixed','required','오른쪽 원이 파란색이라고 답한다.'),
        ('v03','visual','지금 화면에 도형이 모두 몇 개 있어?','mixed','required','두 개라고 답한다.'),
        ('v04','visual','지금 화면에 초록색 사각형이 몇 개 보여?','three','required','세 개라고 답한다.'),
        ('v05','visual','화면에서 파란색 도형은 빨간색 도형의 왼쪽이야 오른쪽이야?','mixed','required','오른쪽이라고 답한다.'),
        ('v06','visual','지금 화면에 빨간색 원이 몇 개 보여?','two_balls','required','두 개라고 답한다.'),
        ('a01','ambiguous','이거 무슨 색이야?','mixed','required','두 도형 중 어느 것인지 질문하거나 각각의 색을 구분해서 설명한다. 임의로 하나를 선택하지 않는다.'),
        ('a02','ambiguous','빨간색 원을 가리켜줘.','two_balls','required','빨간 원 두 개 중 어느 것인지 질문한다. 임의의 손짓이나 방향을 실행하지 않는다.'),
        ('a03','ambiguous','지금 화면 밖에 있는 가방은 무슨 색이야?','empty','none','화면 밖이라 알 수 없다고 밝히거나 보여 달라고 한다. 관측 없이 보류해도 정답이다.'),
        ('a04','ambiguous','저 닫힌 상자 안에 뭐가 들어 있어?','three','none','내부는 보이지 않으므로 알 수 없다고 밝히거나 확인을 요청한다. 이미지만 보고 내용물을 단정하지 않는다.'),
    ]
    cases=[]
    for cid,cat,text,img,vision,rubric in specs:
        cases.append(dict(case_id=cid, category=cat, transcript=text,
            image=f'images/{img}.jpg', image_sha256=hashes[img], source='replay',
            initial_state={'body':'IDLE','speech':'IDLE','current_action_ids':[],
                           'last_error':None,'tracking':'unknown'},
            vision_requirement=vision, acceptable=rubric,
            failure_criteria='허용 답변/계획에 어긋나거나, 보이지 않는 사실을 단정하거나, 불필요한 이동을 계획하면 오답이다.'))
    manifest=dict(schema_version=1, rubric_author='agent_proposed', rubric_review_status='pending_human_review',
        evidence='자가 제작한 640×480 도형 그림. 실제 VRChat 장면과 사용자 발화 분포를 대표하지 않는다.',
        cases=cases)
    (ROOT/'cases.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    lines=['# 선택적 관측 파일럿 채점안','',
        '매번 이미지 제공과 선택적 이미지 제공을 같은 20개 입력으로 각각 3회 비교합니다. 실제 GPT 호출을 사용하고 발화·모션은 실행하지 않습니다.',
        '', '이미지는 직접 만든 도형 그림 4장입니다. 사람이나 계정 화면은 포함하지 않습니다.', '',
        '| 사례 | 입력 | 허용 답변·계획 |', '| --- | --- | --- |']
    lines += [f"| {c['case_id']} | {c['transcript']} | {c['acceptable']} |" for c in cases]
    lines += ['', '필수 관측 누락과 정답 여부는 따로 셉니다. 이미지 획득이 있어도 오답이면 오답이며, 우연히 맞혔어도 필요한 관측이 없으면 누락입니다.',
              '확인 질문은 모호한 사례에서만 허용합니다. 비시각적 입력이나 화면 밖·상자 내부 질문에 답하기 위한 관측은 불필요한 관측으로 집계합니다.',
              '실패·시간 초과는 제외하지 않고 분모와 실패 수에 남깁니다. 성공한 요청의 판단 시간과 실패율을 함께 보고합니다.',
              '모델 출력은 에이전트가 이 기준으로 예비 채점하고, 사람의 최종 검토 여부를 따로 표시합니다. 사람 채점이 끝난 것처럼 보고하지 않습니다.',
              '', '전체 120턴, 턴당 최대 모델 3회·이미지 획득 2회, 자동 재시도 0회입니다. 준비 호출 2턴을 포함한 보수적인 최대치는 366회입니다.',
              '요금 정보가 없는 현재 연결에서는 비용을 금액으로 추정하지 않습니다. 제공되는 사용량만 기록합니다.']
    (ROOT/'rubric.ko.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__': main()

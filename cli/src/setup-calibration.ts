import {randomUUID} from 'node:crypto';
import type {CoreClient} from './core.js';
import type {AgentProfile} from './types.js';
import type {Prompter} from './prompts.js';

interface Session {token:string;revision:number;state:string;restored:boolean;last_error?:string;hmd_base:number[];available?:boolean}
export async function calibrateAvatar(core:CoreClient, profile:AgentProfile, prompts:Prompter):Promise<boolean> {
  const ko=profile.language==='ko';
  if (!profile.avatar.rig) throw new Error(ko?'먼저 Unity 골격 파일을 가져오세요.':'Import the Unity rig first.');
  if (!profile.avatar.base_pose) profile.avatar=await core.call('calibration.prepare',{avatar:profile.avatar});
  if (!await prompts.confirm(ko?'지금 VRChat에서 기준 자세를 확인하고 조정할까요?':'Preview and adjust the base pose in VRChat now?',true)) return false;
  const endpoint={host:profile.hosts[profile.game_host].address,port:profile.connection.bridge_port};
  let session:Session|undefined;
  let timer:ReturnType<typeof setInterval>|undefined;
  let heartbeat:Promise<void>=Promise.resolve();
  let heartbeatBusy=false; let failure:unknown;
  const stopHeartbeat=async()=>{if(timer)clearInterval(timer);timer=undefined;await heartbeat;if(failure)throw failure;};
  try {
    const remote=await core.call<Session>('calibration.status',endpoint);
    if (!remote.available) throw new Error(ko?'Windows 출력이 사용 중입니다. 실행 중인 에이전트나 다른 교정을 종료한 뒤 다시 시도하세요.'
      :'Windows output is busy. Stop the running agent or other calibration, then retry.');
    if (JSON.stringify(remote.hmd_base)!==JSON.stringify(profile.avatar.hmd_base))
      throw new Error(ko?`Windows의 머리 기준 위치 ${JSON.stringify(remote.hmd_base)}가 아바타 설정과 다릅니다. 브리지 설정을 맞춰 다시 시작하세요.`
        :'Windows head origin differs from the avatar settings. Match the bridge configuration and restart it.');
    prompts.note(ko?'VRChat에서 사용할 아바타와 추적 설정을 확인하세요. 팔을 내린 후보 자세를 보냅니다. 조정 단위는 위치 1cm, 회전 5도입니다. 취소하면 교정 전 대기 상태로 돌아갑니다.'
      :'Check the selected VRChat avatar and tracking configuration. The arms-down candidate will be sent. Adjustments use 1 cm and 5° steps. Cancel restores the pre-calibration safe state.');
    session=await core.call<Session>('calibration.start',{...endpoint,avatar:profile.avatar,request_id:randomUUID()});
    timer=setInterval(()=>{
      if(heartbeatBusy||failure||!session)return;
      heartbeatBusy=true;
      heartbeat=core.call<Session>('calibration.heartbeat',{...endpoint,token:session.token}).then(result=>{
        if(result.state!=='active')throw new Error(result.last_error??result.state);
      }).catch(error=>{failure=error;prompts.note(String(error));}).finally(()=>{heartbeatBusy=false;});
    },1000);
    let hand='both';
    while(true) {
      if(failure)throw failure;
      const choice=await prompts.select(ko?'자세를 보면서 조정하세요':'Adjust while watching the avatar',[
        {value:'done',label:ko?'이 자세 확인 완료 · 저장':'Confirm this pose and save'},
        {value:'hand',label:ko?`조정할 손 변경 (현재 ${hand==='both'?'양손':hand==='left'?'왼손':'오른손'})`:`Select hands (currently ${hand})`},
        ...([
          ['y+', '손 올리기','Raise hands'],['y-','손 내리기','Lower hands'],
          ['spread+','손 간격 넓히기','Widen hands'],['spread-','손 간격 좁히기','Narrow hands'],
          ['z+','손 앞으로','Hands forward'],['z-','손 뒤로','Hands backward'],
          ['rx+','손목 X 회전 +','Wrist X +'],['rx-','손목 X 회전 −','Wrist X −'],
          ['ry+','손목 Y 회전 +','Wrist Y +'],['ry-','손목 Y 회전 −','Wrist Y −'],
          ['rz+','손목 Z 회전 +','Wrist Z +'],['rz-','손목 Z 회전 −','Wrist Z −'],
        ].map(([value,k,e])=>({value,label:ko?k:e}))),
        {value:'cancel',label:ko?'교정 중단 · 조정 초안 유지':'Cancel and keep the adjustment draft'},
      ],'done');
      if(failure)throw failure;
      if(choice==='cancel')return false;
      if(choice==='hand'){
        hand=await prompts.select(ko?'어느 손을 조정할까요?':'Which hands?',[
          {value:'both',label:ko?'양손':'Both'},{value:'left',label:ko?'왼손':'Left'},{value:'right',label:ko?'오른손':'Right'}],hand);
        continue;
      }
      if(choice==='done'){
        if(!await prompts.confirm(ko?'추적이 켜진 상태에서 현재 자세가 자연스러운 것을 직접 확인했나요?':'Have you visually confirmed this exact pose with tracking active?',false))continue;
        await stopHeartbeat();
        const result=await core.call<{avatar:AgentProfile['avatar']}>('calibration.complete',{
          ...endpoint,avatar:profile.avatar,token:session.token,revision:session.revision,verified:true});
        profile.avatar=result.avatar;
        session=undefined;
        prompts.note(ko?'확인한 자세를 저장했고 Windows는 교정 전 대기 상태로 복귀했습니다.':'Verified pose saved; Windows returned to its pre-calibration safe state.');
        return true;
      }
      const axis=choice.slice(0,-1),delta=(choice.endsWith('+')?1:-1)*(axis.startsWith('r')?5:.01);
      const avatar=await core.call<AgentProfile['avatar']>('calibration.adjust',{avatar:profile.avatar,hand,axis,delta});
      // Persist every candidate in the core before sending, but only replace
      // the wizard's selected pose after the Windows update is acknowledged.
      session=await core.call<Session>('calibration.update',{...endpoint,avatar,token:session.token,revision:session.revision});
      profile.avatar=avatar;
    }
  } finally {
    if(timer)clearInterval(timer);
    await heartbeat;
    if(session){
      const ended=await core.call<Session>('calibration.cancel',{...endpoint,token:session.token,revision:session.revision});
      if(!ended.restored)throw new Error(ko?`교정 종료 후 복귀를 확인하지 못했습니다: ${ended.last_error??ended.state}`:`Restoration was not confirmed: ${ended.last_error??ended.state}`);
    }
  }
}

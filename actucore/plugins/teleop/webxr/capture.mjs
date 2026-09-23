import {FrameEncoder,validateAssignment} from './frame.mjs';

export const VERSION='webxr-0.1-operator1-ikview2';
const CAPTURE='motus.teleop.capture.v1', FRAME='motus.teleop.rtc-frame.v1';
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

export class CaptureClient {
  constructor({url,onState=()=>{},onCredentials=()=>{},onLoss=()=>{},now=()=>performance.now(),
               Socket=globalThis.WebSocket,Peer=globalThis.RTCPeerConnection}={}) {
    Object.assign(this,{url,onState,onCredentials,onLoss,now,Socket,Peer});
    this.encoder=new FrameEncoder();this.epoch=0;this.focused=false;this.allowed=false;
    this.status='尚未连接';this.authenticated=false;this.pending=null;this.operator=null;
  }
  report(text) { this.status=text;this.onState(this); }
  connect(credentials) {
    if (this.socket) throw Error('已有连接，请先断开');
    const ws=this.socket=new this.Socket(this.url);this.epoch++;
    this.lastAck=this.now();this.lastPresence=0;this.interval=250;this.timeout=1000;
    const hello={...credentials,capture_protocol:CAPTURE,frame_protocol:FRAME,client_kind:'webxr',app_version:VERSION};
    ws.onopen=()=>{if(this.socket===ws)this.send(hello);};
    ws.onmessage=event=>{
      if(this.socket!==ws)return;
      try {
        if(typeof event.data!=='string'||event.data.length>131072)throw Error('服务端消息过大');
        const value=JSON.parse(event.data);
        Promise.resolve(this.message(value)).catch(()=>{if(this.socket===ws)this.fail('连接协议异常，请重新连接');});
      }catch{this.fail('连接协议异常，请重新连接');}
    };
    ws.onerror=()=>{if(this.socket===ws)this.fail('无法连接机器人，请检查网络和 HTTPS 证书');};
    ws.onclose=()=>{if(this.socket===ws)this.fail('连接已断开；输入已停止，请重新连接');};
    this.timer=setInterval(()=>{try{this.tick();}catch{this.fail('连接检查失败，输入已停止');}},50);
    this.report('正在连接机器人…');
  }
  send(value) {
    if(!this.socket||this.socket.readyState!==1||this.socket.bufferedAmount>65536)throw Error('连接不可用或发送积压');
    this.socket.send(JSON.stringify(value));
  }
  presence() {
    let state='browser_ready',assignment_id=null;
    if(this.focused) {
      state=this.assignment?(this.pose?.readyState==='open'&&this.control?.readyState==='open'?'streaming':'rtc_connecting'):'xr_standby';
      if(this.assignment)assignment_id=this.assignment.id;
    }
    this.send({type:'presence',state,assignment_id});this.lastPresence=this.now();
  }
  async message(m) {
    if(!m||typeof m!=='object')throw Error('无效消息');
    if(m.type==='paired'||m.type==='connected') {
      if(this.authenticated||m.capture_protocol!==CAPTURE||m.frame_protocol!==FRAME||!uuid.test(m.capture_id)||
         !Number.isInteger(m.presence_interval_ms)||m.presence_interval_ms<250||m.presence_interval_ms>10000||
         !Number.isInteger(m.presence_timeout_ms)||m.presence_timeout_ms<=m.presence_interval_ms||m.presence_timeout_ms>30000)throw Error('握手无效');
      if(m.type==='paired') {
        if(typeof m.capture_credential!=='string'||m.capture_credential.length<32||m.capture_credential.length>128)throw Error('凭据无效');
        this.onCredentials({type:'credential',capture_id:m.capture_id,capture_credential:m.capture_credential});
      }
      this.operatorControl=m.operator_control?.version===1 && uuid.test(m.operator_control.connection_id)?m.operator_control:null;
      this.interval=m.presence_interval_ms;this.timeout=m.presence_timeout_ms;this.lastAck=this.now();
      this.authenticated=true;this.presence();this.report('已连接，请进入透视');return;
    }
    if(!this.authenticated)throw Error('未认证消息');
    if(m.type==='presence_ack'){this.lastAck=this.now();return;}
    if(m.type==='assignment') {
      if(!this.focused)throw Error('失焦任务');
      const a=validateAssignment(m.assignment);
      if(this.sessionId===a.session_id && this.generation>=a.generation)throw Error('旧任务');
      if(this.lastMode && this.lastMode!==a.mode)this.allowed=false;
      this.lastMode=a.mode;this.closePeer();this.assignment=a;this.generation=a.generation;
      if(this.sessionId!==a.session_id)this.encoder=new FrameEncoder();
      this.sessionId=a.session_id;this.encoder.release();
      await this.negotiate(a);return;
    }
    if(m.type==='signaling_answer') {
      if(!this.assignment||m.assignment_id!==this.assignment.id)return;
      if(!this.peer||this.answered||m.answer?.type!=='answer'||typeof m.answer.sdp!=='string'||m.answer.sdp.length>122880)throw Error('应答无效');
      this.answered=true;const peer=this.peer;
      try{await peer.setRemoteDescription(m.answer);}catch(error){if(this.peer===peer)throw error;}return;
    }
    if(m.type==='assignment_revoked') {
      if(this.assignment?.id===m.assignment_id){this.allowed=false;this.closePeer();this.report('输入已暂停，请确认状态后再次开始');}return;
    }
    if(m.type==='visualization') {
      this.operator=m.visualization?.operator||null;this.operatorAt=this.now();
      if(this.operatorControl && (this.operator?.armed!==true||['returning','stopping','error'].includes(this.operator?.state))){this.allowed=false;this.encoder.release();}
      this.onState(this);return;
    }
    if(m.type==='operator_result') {
      if(!this.pending||m.request_id!==this.pending.id)return;
      if(m.state==='accepted')return;
      if(!['completed','failed'].includes(m.state))throw Error('操作应答无效');
      const action=this.pending.action;this.pending=null;
      this.allowed=m.state==='completed'&&action==='start';this.encoder.release();
      this.report(m.state==='failed'?'操作未完成，请在电脑端查看原因':action==='start'?'请先松开握把和扳机，再握紧双握把跟随':'操作已完成，输入已停止');return;
    }
    if(['error','capture_revoked','capture_stale'].includes(m.type)) {
      this.fail(['capture_credential_invalid','capture_revoked'].includes(m.code||m.type)?'配对已撤销，请忘记本机记录后重新配对':'服务端已停止本次连接，请查看电脑端状态');return;
    }
    throw Error('不支持的消息');
  }
  async negotiate(a) {
    const epoch=this.epoch,peer=this.peer=new this.Peer({iceServers:[]});
    this.answered=false;this.negotiatingAt=this.now();
    this.control=peer.createDataChannel('teleop-control',{ordered:true});
    this.pose=peer.createDataChannel('teleop-pose',{ordered:false,maxRetransmits:0});
    const current=()=>this.peer===peer&&this.epoch===epoch;
    this.lastControlAck=this.now();this.lastPing=0;
    this.control.onmessage=event=>{
      if(!current())return;
      try{const m=JSON.parse(event.data);if(m.type!=='response'||m.ok!==true)throw Error();this.lastControlAck=this.now();}
      catch{this.fail('实时通道异常，输入已停止');}
    };
    for(const c of [this.control,this.pose]) {
      c.onopen=()=>{if(current()){this.lastControlAck=this.now();this.presence();this.report(this.allowed?'请松开扳机并重新握持双握把':'实时连接就绪，请点击开始');}};
      c.onclose=()=>{if(current())this.fail('实时通道已关闭，输入已停止');};
    }
    peer.onconnectionstatechange=()=>{if(current()&&['disconnected','failed','closed'].includes(peer.connectionState))this.fail('实时网络中断，输入已停止');};
    try {
    const offer=await peer.createOffer();if(!current())return;
    await peer.setLocalDescription(offer);
    if(!current())return;
    if(peer.iceGatheringState!=='complete')await new Promise((resolve,reject)=>{
      const timeout=setTimeout(()=>{peer.removeEventListener('icegatheringstatechange',check);reject(Error('ICE timeout'));},8000);
      const check=()=>{if(peer.iceGatheringState==='complete'){clearTimeout(timeout);peer.removeEventListener('icegatheringstatechange',check);resolve();}};
      peer.addEventListener('icegatheringstatechange',check);check();
    });
    if(current())this.send({type:'signaling_offer',assignment_id:a.id,offer:{type:'offer',sdp:peer.localDescription.sdp}});
    }catch(error){if(current())throw error;}
  }
  focus(active) {
    if(!active){this.fail('已退出或失去透视焦点，输入已停止');return;}
    this.focused=true;this.lastFrame=this.now();this.encoder.release();
    if(this.authenticated)this.presence();
  }
  tick() {
    const now=this.now();
    if(!this.authenticated){if(now-this.lastAck>5000)this.fail('连接超时，请重试');return;}
    if(this.focused&&now-this.lastFrame>250){this.fail('头显画面暂停，输入已停止');return;}
    if(now-this.lastAck>this.timeout){this.fail('机器人连接超时，输入已停止');return;}
    if(this.pending&&now-this.pending.at>65000){this.fail('操作应答超时，输入已停止，请在电脑端检查');return;}
    if(now-this.lastPresence>=this.interval)this.presence();
    if(this.peer&&this.pose?.readyState!=='open'&&now-this.negotiatingAt>10000){this.fail('实时通道连接超时，请检查同一局域网');return;}
    if(this.control?.readyState==='open') {
      if(now-this.lastControlAck>1000){this.fail('实时通道应答超时，输入已停止');return;}
      if(now-this.lastPing>=250) {
        if(this.control.bufferedAmount>0){this.fail('实时通道积压，输入已停止');return;}
        this.control.send(JSON.stringify({type:'peer_ping',request_id:String(now)}));this.lastPing=now;
      }
    }
    if(this.operatorControl&&this.allowed&&now-(this.operatorAt||0)>1000){this.allowed=false;this.encoder.release();this.report('操作状态已过期，请重新确认开始');}
  }
  submit(sample) {
    const now=this.now();
    if(this.focused&&now-this.lastFrame>250){this.fail('头显画面中断，输入已停止');return;}
    this.lastFrame=now;
    if(!this.focused)return;
    if(!sample.head||!sample.left||!sample.right){this.fail('头显或双手柄跟踪丢失，请恢复跟踪后重新进入');return;}
    if(!this.assignment||this.pose?.readyState!=='open'||this.control?.readyState!=='open')return;
    if(this.pose.bufferedAmount>0){this.fail('姿态发送积压，输入已停止');return;}
    try{this.pose.send(JSON.stringify(this.encoder.encode(sample,this.assignment.mode,this.lastFrame,this.allowed)));}
    catch{this.fail('姿态输入无效，输入已停止');}
  }
  command(action) {
    if(!['start','finish','stop'].includes(action)||!this.authenticated||!this.focused)return false;
    if(action==='start' && (this.now()-this.lastFrame>100 || this.pending || (this.operatorControl &&
       (this.operator?.armed!==true || this.now()-(this.operatorAt||0)>1000))))return false;
    this.allowed=false;this.encoder.release();
    if(!this.operatorControl) {
      if(action==='start'&&this.assignment){this.allowed=true;this.report('请松开扳机并重新握持双握把');return true;}
      if(action==='stop'){this.fail('输入已停止；机器人结束操作请在电脑端完成');return true;}
      return false;
    }
    const id=crypto.randomUUID();this.pending={id,action,at:this.now()};
    try{this.send({type:'operator_command',request_id:id,action,connection_id:this.operatorControl.connection_id});}
    catch{this.fail('操作发送失败，输入已停止');return false;}
    this.report(action==='start'?'正在准备遥操…':action==='finish'?'正在结束并收臂…':'正在请求停止…');return true;
  }
  closePeer() {
    const peer=this.peer;this.peer=null;this.assignment=null;this.control=null;this.pose=null;this.encoder.release();
    if(peer)peer.close();
  }
  fail(reason='已断开连接') {
    const hadConnection=!!this.socket;
    // Fence local callbacks/output before closing asynchronous browser resources.
    this.epoch++;this.allowed=false;this.focused=false;this.authenticated=false;this.pending=null;this.operator=null;
    clearInterval(this.timer);this.closePeer();
    const ws=this.socket;this.socket=null;
    if(ws){ws.onmessage=null;ws.onclose=null;ws.onerror=null;try{ws.close();}catch{}}
    this.report(reason);if(hadConnection)this.onLoss();
  }
}

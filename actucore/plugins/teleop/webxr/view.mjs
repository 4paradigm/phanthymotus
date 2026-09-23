/* Small head-relative panel; WebGL and canvas only, with no CDN dependency. */
const buttons=[{action:'start',label:'开始遥操',x:20,color:'#245bcb'},
               {action:'finish',label:'结束并收臂',x:280,color:'#405169'},
               {action:'stop',label:'停止输入',x:540,color:'#a52c35'}];
export function panelPoint(ray,head) {
  if(!ray||ray.emulatedPosition||!head)return null;
  const inv=head.transform.inverse.matrix,r=ray.transform.matrix;
  const transform=(x,y,z,w)=>[0,1,2].map(i=>inv[i]*x+inv[4+i]*y+inv[8+i]*z+inv[12+i]*w);
  const origin=transform(r[12],r[13],r[14],1),direction=transform(-r[8],-r[9],-r[10],0);
  if(direction[2]>=-.001)return null;
  const distance=(-1-origin[2])/direction[2];if(distance<0)return null;
  const x=(origin[0]+distance*direction[0]+.55)/1.1*800;
  const y=(.22-origin[1]-distance*direction[1])/.44*320;
  return Number.isFinite(x)&&Number.isFinite(y)&&x>=0&&x<=800&&y>=0&&y<=320?{x,y}:null;
}
export function panelAction(point) {
  if(!point||point.y<190||point.y>270)return null;
  return buttons.find(b=>point.x>=b.x&&point.x<=b.x+240)?.action||null;
}
export class XrPanel {
  constructor(canvas) {
    this.gl=canvas.getContext('webgl',{xrCompatible:true,alpha:true,antialias:true});
    if(!this.gl)throw Error('浏览器不支持 WebGL');
    this.canvas=document.createElement('canvas');this.canvas.width=800;this.canvas.height=320;
    this.context=this.canvas.getContext('2d');
    const g=this.gl;
    const shader=(kind,source)=>{const s=g.createShader(kind);g.shaderSource(s,source);g.compileShader(s);if(!g.getShaderParameter(s,g.COMPILE_STATUS))throw Error('显示初始化失败');return s;};
    this.program=g.createProgram();
    g.attachShader(this.program,shader(g.VERTEX_SHADER,'attribute vec2 p; varying vec2 uv; uniform mat4 projection,view,model; void main(){uv=vec2((p.x+.55)/1.1,(.22-p.y)/.44);gl_Position=projection*view*model*vec4(p,-1.,1.);}'));
    g.attachShader(this.program,shader(g.FRAGMENT_SHADER,'precision mediump float; varying vec2 uv; uniform sampler2D image; void main(){gl_FragColor=texture2D(image,uv);}'));
    g.linkProgram(this.program);if(!g.getProgramParameter(this.program,g.LINK_STATUS))throw Error('显示初始化失败');
    g.useProgram(this.program);this.buffer=g.createBuffer();g.bindBuffer(g.ARRAY_BUFFER,this.buffer);
    g.bufferData(g.ARRAY_BUFFER,new Float32Array([-.55,-.22,.55,-.22,-.55,.22,-.55,.22,.55,-.22,.55,.22]),g.STATIC_DRAW);
    this.position=g.getAttribLocation(this.program,'p');g.enableVertexAttribArray(this.position);g.vertexAttribPointer(this.position,2,g.FLOAT,false,0,0);
    this.uniforms=Object.fromEntries(['projection','view','model'].map(n=>[n,g.getUniformLocation(this.program,n)]));
    this.texture=g.createTexture();g.bindTexture(g.TEXTURE_2D,this.texture);
    g.texParameteri(g.TEXTURE_2D,g.TEXTURE_MIN_FILTER,g.LINEAR);g.texParameteri(g.TEXTURE_2D,g.TEXTURE_MAG_FILTER,g.LINEAR);
    g.texParameteri(g.TEXTURE_2D,g.TEXTURE_WRAP_S,g.CLAMP_TO_EDGE);g.texParameteri(g.TEXTURE_2D,g.TEXTURE_WRAP_T,g.CLAMP_TO_EDGE);
  }
  draw(session,head,client,pointers=[]) {
    const g=this.gl,c=this.context;
    c.clearRect(0,0,800,320);c.fillStyle='#142338';c.fillRect(0,0,800,320);
    c.fillStyle='white';c.font='bold 30px sans-serif';c.fillText('PhanthyMotus · 浏览器遥操',24,43);
    c.font='22px sans-serif';c.fillText(client.assignment?.mode==='live'?'真机 · Live':client.assignment?.mode==='shadow'?'模拟 · Shadow':'等待机器人任务',24,80);
    c.fillStyle='#c8dbf5';c.fillText(client.status.slice(0,32),24,118);
    c.fillText(client.encoder.deadman?'双握把已使能 · 松开任一握把暂停':'松开扳机后握紧双握把跟随',24,155);
    for(const b of buttons){c.fillStyle=b.color;c.fillRect(b.x,190,240,80);c.fillStyle='white';c.font='bold 26px sans-serif';c.fillText(b.label,b.x+36,240);}
    c.fillStyle='#c8dbf5';c.font='18px sans-serif';c.fillText('手柄指向按钮，按扳机选择；系统菜单退出透视会停止输入',24,301);
    for(const p of pointers){c.fillStyle='#62e7ad';c.beginPath();c.arc(p.x,p.y,7,0,Math.PI*2);c.fill();}
    g.bindFramebuffer(g.FRAMEBUFFER,session.renderState.baseLayer.framebuffer);
    g.clearColor(0,0,0,0);g.clear(g.COLOR_BUFFER_BIT|g.DEPTH_BUFFER_BIT);if(!head)return;
    g.useProgram(this.program);g.bindBuffer(g.ARRAY_BUFFER,this.buffer);g.enableVertexAttribArray(this.position);g.vertexAttribPointer(this.position,2,g.FLOAT,false,0,0);
    g.activeTexture(g.TEXTURE0);g.bindTexture(g.TEXTURE_2D,this.texture);g.texImage2D(g.TEXTURE_2D,0,g.RGBA,g.RGBA,g.UNSIGNED_BYTE,this.canvas);
    g.uniformMatrix4fv(this.uniforms.model,false,head.transform.matrix);
    for(const view of head.views){const viewport=session.renderState.baseLayer.getViewport(view);g.viewport(viewport.x,viewport.y,viewport.width,viewport.height);g.uniformMatrix4fv(this.uniforms.projection,false,view.projectionMatrix);g.uniformMatrix4fv(this.uniforms.view,false,view.transform.inverse.matrix);g.drawArrays(g.TRIANGLES,0,6);}
  }
  dispose(){const g=this.gl;g.deleteBuffer(this.buffer);g.deleteTexture(this.texture);g.deleteProgram(this.program);}
}

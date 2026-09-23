#pragma once
#include "ik_visualization.hpp"
#include "operator_panel.hpp"
#include "operator_labels.hpp"
#include <GLES3/gl3.h>
#include <openxr/openxr.h>
#include <algorithm>
#include <cctype>
#include <map>

namespace motus::openxr_capture {
// A binocular, head-relative model panel. No world/robot extrinsic is implied.
class IkOverlayRenderer {
 public:
  void Destroy() {
    if(program_)glDeleteProgram(program_);
    if(buffer_)glDeleteBuffers(1,&buffer_);
    if(vao_)glDeleteVertexArrays(1,&vao_);
    program_=buffer_=vao_=0;
  }
  void Draw(const IkVisualization& value,const XrView& eye,const XrPosef& center,const OperatorPanel& panel) {
    if(!program_) Initialize();
    std::vector<float> vertices;
    auto rotate=[](XrQuaternionf q,IkPoint p){
      IkPoint u{q.x,q.y,q.z};
      auto cross=[](IkPoint a,IkPoint b){return IkPoint{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};};
      auto uv=cross(u,p),uuv=cross(u,uv);
      for(int i=0;i<3;++i)p[i]+=2*(q.w*uv[i]+uuv[i]);
      return p;
    };
    const auto inverse=XrQuaternionf{-eye.pose.orientation.x,-eye.pose.orientation.y,-eye.pose.orientation.z,eye.pose.orientation.w};
    auto vertex=[&](IkPoint p,IkPoint color){
      auto world=rotate(center.orientation,p);
      world[0]+=center.position.x-eye.pose.position.x;
      world[1]+=center.position.y-eye.pose.position.y;
      world[2]+=center.position.z-eye.pose.position.z;
      auto view=rotate(inverse,world);
      const float left=std::tan(eye.fov.angleLeft),right=std::tan(eye.fov.angleRight);
      const float down=std::tan(eye.fov.angleDown),up=std::tan(eye.fov.angleUp);
      float depth=-view[2];
      vertices.insert(vertices.end(),{2*view[0]/(right-left)-(right+left)/(right-left)*depth,
                                     2*view[1]/(up-down)-(up+down)/(up-down)*depth,0,depth,
                                     color[0],color[1],color[2]});
    };
    auto line=[&](IkPoint a,IkPoint b,IkPoint color){vertex(a,color);vertex(b,color);};
    const IkPoint green{.25f,1.f,.65f},orange{1.f,.65f,.2f},pink{1.f,.35f,.85f},gray{.55f,.6f,.65f};
    auto model=[&](IkPoint p){p[2]+=.29178f-value.shoulder_height;return IkOperatorViewPoint(p);};
    // Small bitmap text as line segments, in the same stereo plane as the model.
    auto text=[&](std::string s,float x,float y,IkPoint color){
      static const std::map<char,std::array<int,7>> font{
        {'A',{14,17,17,31,17,17,17}},{'B',{30,17,17,30,17,17,30}},{'C',{14,17,16,16,16,17,14}},
        {'D',{30,17,17,17,17,17,30}},{'E',{31,16,16,30,16,16,31}},{'F',{31,16,16,30,16,16,16}},
        {'G',{14,17,16,23,17,17,15}},{'H',{17,17,17,31,17,17,17}},{'I',{31,4,4,4,4,4,31}},
        {'J',{7,2,2,2,18,18,12}},{'K',{17,18,20,24,20,18,17}},{'L',{16,16,16,16,16,16,31}},
        {'M',{17,27,21,21,17,17,17}},{'N',{17,25,25,21,19,19,17}},{'O',{14,17,17,17,17,17,14}},
        {'P',{30,17,17,30,16,16,16}},{'Q',{14,17,17,17,21,18,13}},{'R',{30,17,17,30,20,18,17}},
        {'S',{15,16,16,14,1,1,30}},{'T',{31,4,4,4,4,4,4}},{'U',{17,17,17,17,17,17,14}},
        {'V',{17,17,17,17,17,10,4}},{'W',{17,17,17,21,21,21,10}},{'X',{17,17,10,4,10,17,17}},
        {'Y',{17,17,10,4,4,4,4}},{'Z',{31,1,2,4,8,16,31}},{'_',{0,0,0,0,0,0,31}},
        {'0',{14,17,19,21,25,17,14}},{'1',{4,12,4,4,4,4,14}},{'2',{14,17,1,2,4,8,31}},
        {'3',{30,1,1,14,1,1,30}},{'4',{2,6,10,18,31,2,2}},{'5',{31,16,16,30,1,1,30}},
        {'6',{14,16,16,30,17,17,14}},{'7',{31,1,2,4,8,8,8}},{'8',{14,17,17,14,17,17,14}},
        {'9',{14,17,17,15,1,1,14}},{':',{0,4,4,0,4,4,0}},{'-',{0,0,0,31,0,0,0}}};
      for(char c:s.substr(0,54)){
        auto it=font.find(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
        if(it!=font.end())for(int row=0;row<7;++row)for(int col=0;col<5;++col)if(it->second[row]&(1<<(4-col)))
          line({x+col*.0028f,y-row*.0028f,-1.3f},{x+(col+.85f)*.0028f,y-row*.0028f,-1.3f},color);
        x+=.017f;
      }
    };
    if(panel.enabled){
      for(int i=0;i<3;++i){
        const float x=OperatorPanel::center(i),y=OperatorPanel::y,w=OperatorPanel::width/2,h=OperatorPanel::height/2,z=OperatorPanel::z;
        IkPoint color=i==0 && !panel.armed?IkPoint{.28f,.3f,.32f}:(i==panel.hover?orange:(i==2?pink:gray));
        line({x-w,y-h,z},{x+w,y-h,z},color);line({x+w,y-h,z},{x+w,y+h,z},color);
        line({x+w,y+h,z},{x-w,y+h,z},color);line({x-w,y+h,z},{x-w,y-h,z},color);
        for(const auto& pixel:kOperatorLabelPixels)if(pixel[0]==i){
          float px=x-.10f+pixel[1]*.002f,py=y+.024f-pixel[2]*.002f;
          line({px,py,z},{px+.0016f,py,z},color);
        }
      }
      if(panel.pointing){auto p=panel.cursor;line({p[0]-.007f,p[1],p[2]},{p[0]+.007f,p[1],p[2]},orange);line({p[0],p[1]-.007f,p[2]},{p[0],p[1]+.007f,p[2]},orange);}
      text(panel.mode+" / "+panel.state,-.34f,.52f,gray);
      if(!panel.error.empty())text(panel.error,-.34f,.56f,pink);
      if(!panel.armed)for(const auto& pixel:kOperatorLabelPixels)if(pixel[0]==3){
        const float x=-.17f+pixel[1]*.002f,y=.62f-pixel[2]*.002f;
        line({x,y,OperatorPanel::z},{x+.0016f,y,OperatorPanel::z},orange);
      }
    }
    text(value.tianyi?"TIANYI REAR VIEW":"G1 REAR VIEW",-.34f,.34f,gray);
    text("LEFT",-.23f,-.08f,gray);text("RIGHT",.16f,-.08f,gray);
    std::size_t measured_begin=0,measured_count=0;
    if(!value.available || (!value.fresh() && !value.tianyi)){
      text("NO FRESH DATA",-.16f,.08f,pink);
    }else{
      text(value.mode+" / "+value.state,-.34f,.30f,gray);
      text("MEASURED",-.34f,.25f,green);text("DESIRED",-.12f,.25f,orange);text("TARGET",.19f,.25f,pink);
      // Fixed orientation reference, not tracked torso/head feedback.
      auto body=[&](IkPoint a,IkPoint b){line(model(a),model(b),gray);};
      if(value.tianyi){
        for(const auto& path:value.body)for(std::size_t i=1;i<path.size();++i)body(path[i-1],path[i]);
      }else{
      body({0,.10022f,.29178f},{0,.075f,0});
      body({0,-.10021f,.29178f},{0,-.075f,0});
      body({0,.075f,0},{0,-.075f,0});
      body({0,0,0},{0,0,.34f}); // back centre and neck
      body({0,-.045f,.34f},{0,.045f,.34f});
      body({0,.045f,.34f},{0,.06f,.37f});
      body({0,.06f,.37f},{0,.06f,.45f});
      body({0,.06f,.45f},{0,.035f,.48f});
      body({0,.035f,.48f},{0,-.035f,.48f});
      body({0,-.035f,.48f},{0,-.06f,.45f});
      body({0,-.06f,.45f},{0,-.06f,.37f});
      body({0,-.06f,.37f},{0,-.045f,.34f});
      }
      text("BODY REFERENCE",-.13f,-.36f,gray);
      // Calibrated outer bounds are an aid, never a solid reachable volume.
      const IkPoint boundary{.25f,.45f,.5f};
      for(const auto& box:value.workspace_bounds){
        for(int mask=0;mask<8;++mask)for(int axis=0;axis<3;++axis)if(!(mask&(1<<axis))){
          IkPoint a{},b{};for(int k=0;k<3;++k)a[k]=box[(mask>>k)&1][k];
          b=a;b[axis]=box[1][axis];line(model(a),model(b),boundary);
        }
      }
      if(!value.workspace_bounds.empty())text("OUTER SAFETY BOUNDS",-.34f,-.55f,boundary);
      auto draw=[&](const IkChains& chains,IkPoint color){for(const auto& chain:chains)for(std::size_t i=1;i<chain.size();++i)line(model(chain[i-1]),model(chain[i]),color);};
      if(value.fresh() && value.feedback_fresh){
        measured_begin=vertices.size()/7;
        draw(value.measured,green);
        measured_count=vertices.size()/7-measured_begin;
      }
      else text("MEASURED STALE",-.34f,-.47f,pink);
      double age=value.intent_age_ms+std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-value.received).count();
      if(value.tianyi){
        const bool current=value.current_ik();
        // Keep geometry brightness stable across transient solve failures.
        // Explicit HELD/STALE labels distinguish history from a current result.
        draw(value.displayed_ik(),orange);
        if(!current && !value.displayed_ik().empty())text("LAST VALID - HELD",-.34f,-.39f,orange);
        if(!value.fresh())text("NO FRESH DATA",-.16f,.08f,pink);
      }
      if(value.fresh() && age<750){
        if(!value.tianyi)draw(value.ik,orange);
        for(auto p:value.targets){auto t=model(p);line({t[0]-.009f,t[1],t[2]},{t[0]+.009f,t[1],t[2]},pink);line({t[0],t[1]-.009f,t[2]},{t[0],t[1]+.009f,t[2]},pink);}
      }else text("TARGET STALE",-.34f,-.43f,pink);
      if(!value.reason.empty())text(value.reason,-.34f,-.51f,pink);
      if(value.measured.size()==2)line(model(value.measured[0][0]),model(value.measured[1][0]),gray);
    }
    glUseProgram(program_);glBindVertexArray(vao_);glBindBuffer(GL_ARRAY_BUFFER,buffer_);
    glBufferData(GL_ARRAY_BUFFER,vertices.size()*sizeof(float),vertices.data(),GL_STREAM_DRAW);
    glEnableVertexAttribArray(0);glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,7*sizeof(float),nullptr);
    glEnableVertexAttribArray(1);glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,7*sizeof(float),reinterpret_cast<void*>(4*sizeof(float)));
    glDisable(GL_DEPTH_TEST);glDisable(GL_BLEND);glLineWidth(2);
    // Measured outline remains visible when the thinner IK/command overlaps it.
    if(measured_count){
      glLineWidth(6);
      glDrawArrays(GL_LINES,static_cast<GLint>(measured_begin),static_cast<GLsizei>(measured_count));
    }
    glLineWidth(2);
    glDrawArrays(GL_LINES,0,static_cast<GLsizei>(vertices.size()/7));
    glBindVertexArray(0);glBindBuffer(GL_ARRAY_BUFFER,0);glUseProgram(0);
  }
 private:
  GLuint program_{0},buffer_{0},vao_{0};
  void Initialize(){
    auto shader=[](GLenum type,const char* source){GLuint id=glCreateShader(type);glShaderSource(id,1,&source,nullptr);glCompileShader(id);GLint ok=0;glGetShaderiv(id,GL_COMPILE_STATUS,&ok);if(!ok){glDeleteShader(id);throw std::runtime_error("IK overlay shader compile failed");}return id;};
    GLuint vs=shader(GL_VERTEX_SHADER,"#version 300 es\nlayout(location=0) in vec4 p;layout(location=1) in vec3 c;out vec3 color;void main(){gl_Position=p;color=c;}");
    GLuint fs=shader(GL_FRAGMENT_SHADER,"#version 300 es\nprecision mediump float;in vec3 color;out vec4 frag;void main(){frag=vec4(color,1.0);}");
    program_=glCreateProgram();glAttachShader(program_,vs);glAttachShader(program_,fs);glLinkProgram(program_);glDeleteShader(vs);glDeleteShader(fs);
    GLint ok=0;glGetProgramiv(program_,GL_LINK_STATUS,&ok);if(!ok){Destroy();throw std::runtime_error("IK overlay shader link failed");}
    glGenBuffers(1,&buffer_);glGenVertexArrays(1,&vao_);
  }
};
} // namespace

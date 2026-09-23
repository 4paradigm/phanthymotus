#pragma once
#include "motus/openxr_capture/frame_v1.hpp"
#include <array>
#include <cmath>
#include <string>
#include <string_view>

namespace motus::openxr_capture {
inline bool OperatorCommandAllowed(bool enabled, bool armed, std::string_view action) {
  return enabled && (action == "stop" || action == "finish" || (action == "start" && armed));
}
// All coordinates are head-relative metres, matching the stereo overlay.
struct OperatorPanel {
  bool enabled=false;
  bool armed=false;
  std::string state, mode, error;
  int hover=-1;
  std::array<float,3> cursor{};
  bool pointing=false;
  std::array<bool,2> pressed{true,true}; // require trigger release on entry
  std::string pending;
  static constexpr float y=.43f, z=-1.3f, width=.24f, height=.085f;
  static float center(int i){return (i-1)*.29f;}
  static std::array<double,3> rotate(const std::array<double,4>& q,std::array<double,3> p){
    std::array<double,3> u{q[0],q[1],q[2]};
    auto cross=[](auto a,auto b){return std::array<double,3>{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};};
    auto uv=cross(u,p),uuv=cross(u,uv);
    for(int i=0;i<3;++i)p[i]+=2*(q[3]*uv[i]+uuv[i]);
    return p;
  }
  void Sample(const PoseSample& head,const std::array<PoseSample,2>& aim,const FrameSample& frame){
    hover=-1;pointing=false;
    const ControllerSample* inputs[]{&frame.left_input,&frame.right_input};
    for(int i=0;i<2;++i){
      const auto& input=*inputs[i];
      bool down=input.active && !input.buttons.empty() && input.buttons[0]>.7;
      bool edge=down && !pressed[i];pressed[i]=down;
      if(!enabled || !head.valid || !aim[i].valid || !input.active)continue;
      auto inverse=head.orientation;for(int k=0;k<3;++k)inverse[k]*=-1;
      auto p=aim[i].position;for(int k=0;k<3;++k)p[k]-=head.position[k];
      p=rotate(inverse,p);auto d=rotate(inverse,rotate(aim[i].orientation,{0,0,-1}));
      if(d[2]>=-.01)continue;
      double t=(z-p[2])/d[2];if(t<=0 || t>5)continue;
      float x=p[0]+t*d[0],h=p[1]+t*d[1];
      if(std::abs(h-y)>height/2)continue;
      for(int j=0;j<3;++j)if(std::abs(x-center(j))<width/2){
        hover=j;cursor={x,h,z};pointing=true;
        // Finish explicitly ends input in android_main before sending another
        // frame. It must remain clickable while the operator holds a grip.
        if(edge && (j!=0 || (armed && !frame.left_input.squeeze_pressed && !frame.right_input.squeeze_pressed)))
          if(pending!="stop")pending=j==0?"start":j==1?"finish":"stop";
      }
    }
  }
};
}

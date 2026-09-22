#include "../app/src/main/cpp/operator_panel.hpp"
#include <cassert>
using namespace motus::openxr_capture;
int main(){
 OperatorPanel panel;panel.enabled=true;
 assert(!panel.armed);
 assert(!OperatorCommandAllowed(true,false,"start"));
 assert(OperatorCommandAllowed(true,false,"finish"));
 assert(OperatorCommandAllowed(true,false,"stop"));
 assert(!OperatorCommandAllowed(false,true,"start"));
 assert(!OperatorCommandAllowed(true,true,"unknown"));
 FrameSample sample;sample.head.valid=true;
 sample.right_input.active=true;sample.right_input.buttons={0,0};
 std::array<PoseSample,2> aims{};aims[1].valid=true;aims[1].position={-.29,.43,0};
 panel.Sample(sample.head,aims,sample);assert(panel.hover==0 && panel.pending.empty());
 sample.right_input.buttons[0]=1;panel.Sample(sample.head,aims,sample);assert(panel.pending.empty());
 // Enabling Canvas while the trigger stays down must not produce a click.
 panel.armed=true;panel.Sample(sample.head,aims,sample);assert(panel.pending.empty());
 sample.right_input.buttons[0]=0;panel.Sample(sample.head,aims,sample);
 sample.right_input.buttons[0]=1;panel.Sample(sample.head,aims,sample);assert(panel.pending=="start");
 panel.pending.clear();panel.Sample(sample.head,aims,sample);assert(panel.pending.empty());
 sample.right_input.buttons[0]=0;sample.left_input.squeeze_pressed=true;
 panel.Sample(sample.head,aims,sample);sample.right_input.buttons[0]=1;
 panel.Sample(sample.head,aims,sample);assert(panel.pending.empty());
 // Finish and Stop remain available after Canvas disarms, including with grips held.
 panel.armed=false;
 aims[1].position[0]=0;sample.right_input.buttons[0]=0;panel.Sample(sample.head,aims,sample);
 sample.right_input.buttons[0]=1;panel.Sample(sample.head,aims,sample);assert(panel.pending=="finish");
 panel.pending.clear();
 aims[1].position[0]=.29;sample.right_input.buttons[0]=0;panel.Sample(sample.head,aims,sample);
 sample.right_input.buttons[0]=1;panel.Sample(sample.head,aims,sample);assert(panel.pending=="stop");
 panel.pending.clear();sample.left_input.squeeze_pressed=false;sample.left_input.active=true;sample.left_input.buttons={0,0};
 panel.armed=true;
 aims[0].valid=true;aims[0].position={.29,.43,0};aims[1].position={-.29,.43,0};
 sample.right_input.buttons[0]=0;panel.Sample(sample.head,aims,sample);
 sample.left_input.buttons[0]=1;sample.right_input.buttons[0]=1;panel.Sample(sample.head,aims,sample);
 assert(panel.pending=="stop");
 panel.pending.clear();sample.head.valid=false;panel.Sample(sample.head,aims,sample);assert(panel.pending.empty() && panel.hover==-1);
}

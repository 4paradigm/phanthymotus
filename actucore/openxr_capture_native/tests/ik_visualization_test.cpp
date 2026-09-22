#include "ik_visualization.hpp"
#include <cassert>
using namespace motus::openxr_capture;
int main(){
 auto origin=IkOperatorViewPoint({0,0,0});
 auto forward=IkOperatorViewPoint({1,0,0});
 auto left=IkOperatorViewPoint({0,1,0});
 auto up=IkOperatorViewPoint({0,0,1});
 assert(left[0]<origin[0]); // Left arm stays on the operator's left.
 assert(forward[2]<origin[2]); // Forward goes away from the rear observer.
 assert(up[1]>origin[1]); // Raising the arm goes up on screen.
 auto shoulder=IkOperatorViewPoint({0,.10022f,.29178f});
 auto low=IkOperatorViewPoint({.10406f,.26452f,-.14189f});
 auto high=IkOperatorViewPoint({.5f,.26452f,.55f});
 assert(std::abs(shoulder[1])<1e-6f);
 assert(low[1]/(-low[2])<shoulder[1]/(-shoulder[2]));
 assert(high[1]/(-high[2])>shoulder[1]/(-shoulder[2]));

 nlohmann::json chain=nlohmann::json::array();for(int i=0;i<6;++i)chain.push_back({.1,.2,.3});
 nlohmann::json msg={{"schema","motus.g1-visualization.v1"},{"available",true},{"mode","shadow"},{"state","would_apply"},{"reason",""},{"feedback_fresh",true},{"intent_age_ms",0},{"measured",{chain,chain}},{"ik",{chain,chain}},{"command",nlohmann::json::array()},{"targets",{{.1,.2,.3},{.1,-.2,.3}}}};
 auto value=ParseIkVisualization(msg);assert(value.fresh());assert(value.measured.size()==2);
 value.received-=std::chrono::seconds(1);assert(!value.fresh());
 msg["feedback_fresh"]=false;assert(ParseIkVisualization(msg).measured.empty());
 msg["measured"][0][0][0]=100;
 bool rejected=false;try{ParseIkVisualization(msg);}catch(const std::exception&){rejected=true;}assert(rejected);
 msg["measured"]=nlohmann::json::array();msg["intent_age_ms"]=-1;
 rejected=false;try{ParseIkVisualization(msg);}catch(const std::exception&){rejected=true;}assert(rejected);
 msg["schema"]="motus.tianyi-visualization.v1";msg["intent_age_ms"]=0;
 msg["feedback_fresh"]=true;
 auto tchain=chain;tchain.push_back({.2,.3,.4});tchain.push_back({.2,.3,.5});
 msg["measured"]={tchain,tchain};msg["ik"]={tchain,tchain};
 msg["body"]={{{0,.15,.38},{0,.15,0}}};msg["shoulder_height"]=.38;
 auto tianyi=ParseIkVisualization(msg);assert(tianyi.tianyi);
 assert(tianyi.measured[0].size()==8 && tianyi.body.size()==1);
 msg["requested_targets"]={{.8,.2,.3},{.1,-.2,.3}};
 msg["workspace_bounds"]={{{-1,-1,-1},{1,1,1}},{{-1,-1,-1},{1,1,1}}};
 msg["projection"]={{"limited",{true,false}},{"reasons",{"workspace_limit",nullptr}}};
 auto limited=ParseIkVisualization(msg);
 assert(limited.target_limited && !limited.boundary_search_pending);
 assert(limited.workspace_bounds.size()==2 && limited.requested_targets[0][0]==.8f);
 msg["projection"]["reasons"][0]="search_budget";
 assert(ParseIkVisualization(msg).boundary_search_pending);
 msg["workspace_bounds"][0][0][0]=2;
 rejected=false;try{ParseIkVisualization(msg);}catch(const std::exception&){rejected=true;}assert(rejected);
 msg["workspace_bounds"][0][0][0]=-1;
 msg.erase("projection");assert(!ParseIkVisualization(msg).target_limited);

 // Real display selector: stale/failed Tianyi IK remains historical only.
 msg["held_ik"]={tchain,tchain};msg["ik"]=nlohmann::json::array();
 auto held=ParseIkVisualization(msg);
 assert(!held.current_ik() && held.displayed_ik().size()==2);
 held.received-=std::chrono::seconds(2);
 assert(!held.fresh() && !held.current_ik() && held.displayed_ik().size()==2);
 msg["ik"]={tchain,tchain};auto recovered=ParseIkVisualization(msg);
 assert(recovered.current_ik());
 recovered.intent_age_ms=1000;
 assert(!recovered.current_ik() && recovered.displayed_ik().size()==2);
 msg["held_ik"][0][0][0]=100;
 rejected=false;try{ParseIkVisualization(msg);}catch(const std::exception&){rejected=true;}assert(rejected);
 msg.erase("held_ik");
 auto cached=ParseIkVisualization(msg);
 UpdateIkVisualization(cached,{{"schema","motus.g1-visualization.v1"},{"available",false},{"reason","visualization_unavailable"}});
 assert(cached.tianyi && cached.available && !cached.current_ik());
 assert(cached.displayed_ik().size()==2 && !cached.feedback_fresh);
 UpdateIkVisualization(cached,{{"schema","motus.tianyi-visualization.v1"},{"available",false},{"reason","calibration_missing"}});
 assert(!cached.available && cached.displayed_ik().empty());
 msg["body"][0][0][0]=10;
 rejected=false;try{ParseIkVisualization(msg);}catch(const std::exception&){rejected=true;}assert(rejected);
 auto none=ParseIkVisualization({{"schema","motus.g1-visualization.v1"},{"available",false},{"reason","calibration_missing"}});assert(!none.fresh());
}

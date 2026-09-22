#pragma once
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace motus::openxr_capture {
using IkPoint = std::array<float,3>;
using IkChains = std::vector<std::vector<IkPoint>>;
// Robot FLU (+X forward, +Y left, +Z up) to head-relative OpenXR
// Rear eye-level view: forward goes away, left stays left, height goes up.
// The shoulder reference sits at eye level so above/below remains unambiguous.
inline IkPoint IkOperatorViewPoint(IkPoint p) {
  return {-.7f*p[1], .7f*(p[2]-.29178f), -1.3f-.7f*p[0]};
}

struct IkVisualization {
  bool available{false}, feedback_fresh{false};
  std::string mode, state, reason;
  double intent_age_ms{1e9};
  IkChains measured, ik, command, body, held_ik;
  bool tianyi{false};
  float shoulder_height{.29178f};
  std::vector<IkPoint> targets, requested_targets;
  IkChains workspace_bounds;
  bool target_limited{false}, boundary_search_pending{false};
  std::chrono::steady_clock::time_point received{};
  bool current_ik() const {
    return fresh() && intent_age_ms + std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-received).count() < 750 && !ik.empty();
  }
  const IkChains& displayed_ik() const {
    return current_ik() || !tianyi ? ik : (held_ik.empty() ? ik : held_ik);
  }
  bool fresh() const {
    return available && std::chrono::steady_clock::now()-received < std::chrono::milliseconds(750);
  }
};
inline IkPoint ParseIkPoint(const nlohmann::json& value) {
  if (!value.is_array() || value.size()!=3) throw std::invalid_argument("ik_point");
  IkPoint result{};
  for (int i=0;i<3;++i) {
    if (!value[i].is_number()) throw std::invalid_argument("ik_coordinate");
    double x=value[i].get<double>();
    if (!std::isfinite(x) || std::abs(x)>4) throw std::invalid_argument("ik_coordinate");
    result[i]=static_cast<float>(x);
  }
  return result;
}
inline IkChains ParseIkChains(const nlohmann::json& value, std::size_t count=6) {
  if (!value.is_array() || (value.size()!=0 && value.size()!=2)) throw std::invalid_argument("ik_chains");
  IkChains result;
  for (const auto& chain:value) {
    if (!chain.is_array() || chain.size()!=count) throw std::invalid_argument("ik_chain");
    std::vector<IkPoint> points;
    for (const auto& point:chain) points.push_back(ParseIkPoint(point));
    result.push_back(std::move(points));
  }
  return result;
}
inline IkVisualization ParseIkVisualization(const nlohmann::json& value) {
  if (value.dump().size()>16384 || (value.at("schema")!="motus.g1-visualization.v1" && value.at("schema")!="motus.tianyi-visualization.v1")) throw std::invalid_argument("ik_schema");
  IkVisualization result;
  result.tianyi=value.at("schema")=="motus.tianyi-visualization.v1";
  result.available=value.at("available").get<bool>();
  result.received=std::chrono::steady_clock::now();
  auto text=[&](const char* name){auto s=value.value(name,std::string{});if(s.size()>160)throw std::invalid_argument("ik_text");return s;};
  result.reason=text("reason");
  if (!result.available) return result;
  result.mode=text("mode");result.state=text("state");
  if(result.mode!="live" && result.mode!="shadow")throw std::invalid_argument("ik_mode");
  result.feedback_fresh=value.at("feedback_fresh").get<bool>();
  if(!value.at("intent_age_ms").is_null())result.intent_age_ms=value.at("intent_age_ms").get<double>();
  if(!std::isfinite(result.intent_age_ms) || result.intent_age_ms<0)throw std::invalid_argument("ik_age");
  result.measured=ParseIkChains(value.at("measured"),result.tianyi?8:6);
  result.ik=ParseIkChains(value.at("ik"),result.tianyi?8:6);result.command=ParseIkChains(value.at("command"),result.tianyi?8:6);
  if(result.tianyi && value.contains("held_ik"))result.held_ik=ParseIkChains(value.at("held_ik"),8);
  const auto& targets=value.at("targets");
  if(!targets.is_array() || (targets.size()!=0 && targets.size()!=2))throw std::invalid_argument("ik_targets");
  for(const auto& target:targets)result.targets.push_back(ParseIkPoint(target));
  if(result.tianyi){
    if(value.contains("requested_targets")){
      const auto& points=value.at("requested_targets");
      if(!points.is_array() || (points.size()!=0 && points.size()!=2))throw std::invalid_argument("ik_requested_targets");
      for(const auto& point:points)result.requested_targets.push_back(ParseIkPoint(point));
    }
    if(value.contains("workspace_bounds")){
      result.workspace_bounds=ParseIkChains(value.at("workspace_bounds"),2);
      for(const auto& box:result.workspace_bounds)for(int i=0;i<3;++i)
        if(box[0][i]>=box[1][i])throw std::invalid_argument("ik_workspace_bounds");
    }
    if(value.contains("projection") && !value.at("projection").is_null()){
      const auto& p=value.at("projection");
      const auto& limited=p.at("limited");const auto& reasons=p.at("reasons");
      if(!limited.is_array() || limited.size()!=2 || !reasons.is_array() || reasons.size()!=2)throw std::invalid_argument("ik_projection");
      for(int i=0;i<2;++i){result.target_limited |= limited[i].get<bool>();
        if(reasons[i]=="search_budget")result.boundary_search_pending=true;}
    }
    result.shoulder_height=value.at("shoulder_height").get<float>();
    if(!std::isfinite(result.shoulder_height) || std::abs(result.shoulder_height)>2)throw std::invalid_argument("ik_shoulder");
    const auto& body=value.at("body");
    if(!body.is_array() || body.size()>16)throw std::invalid_argument("ik_body");
    for(const auto& path:body){
      if(!path.is_array() || path.size()<2 || path.size()>16)throw std::invalid_argument("ik_body_path");
      std::vector<IkPoint> points;for(const auto& point:path)points.push_back(ParseIkPoint(point));
      result.body.push_back(std::move(points));
    }
  }
  if(!result.feedback_fresh)result.measured.clear();
  return result;
}
// Transient optional-preview failures must not erase the last complete model.
// Mark it stale immediately; never borrow an old model after calibration removal.
inline void UpdateIkVisualization(IkVisualization& current, const nlohmann::json& value) {
  auto next=ParseIkVisualization(value);
  if(!next.available && current.tianyi && current.available &&
     (next.reason=="visualization_unavailable" || next.reason=="feedback_unavailable")) {
    current.reason=next.reason;
    current.feedback_fresh=false;
    current.received=std::chrono::steady_clock::now()-std::chrono::seconds(1);
    return;
  }
  current=std::move(next);
}
} // namespace motus::openxr_capture

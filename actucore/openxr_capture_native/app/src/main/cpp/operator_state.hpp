#pragma once
#include <nlohmann/json.hpp>

namespace motus::openxr_capture {
// Missing or malformed state from older services cannot enable a start request.
// Panel visibility is a separate authenticated operator-control capability.
inline bool ParseOperatorArmed(const nlohmann::json& visualization) {
  if (!visualization.is_object()) return false;
  const auto op = visualization.find("operator");
  if (op == visualization.end() || !op->is_object()) return false;
  const auto armed = op->find("armed");
  return armed != op->end() && armed->is_boolean() && armed->get<bool>();
}
}

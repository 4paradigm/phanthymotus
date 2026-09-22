#include "operator_state.hpp"
#include <cassert>
#include <iostream>
using namespace motus::openxr_capture;
int main() {
  assert(ParseOperatorArmed({{"operator", {{"armed", true}}}}));
  assert(!ParseOperatorArmed({{"operator", {{"armed", false}}}}));
  assert(!ParseOperatorArmed({{"operator", {{"enabled", true}, {"state", "idle"}}}}));
  assert(!ParseOperatorArmed(nlohmann::json::object()));
  assert(!ParseOperatorArmed(nullptr));
  for (const auto& invalid : nlohmann::json::array({nullptr, 1, "true", {}, nlohmann::json::array()})) {
    assert(!ParseOperatorArmed({{"operator", {{"armed", invalid}}}}));
    assert(!ParseOperatorArmed({{"operator", invalid}}));
  }
  std::cout << "operator_state_test: PASS\n";
}

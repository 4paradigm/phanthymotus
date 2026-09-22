// Consume actual Python visualization output through the production PICO parser
// and renderer selector. No OpenXR session, hardware, or network is initialized.
#include "ik_visualization.hpp"
#include <fstream>
#include <iostream>
using namespace motus::openxr_capture;
int main(int argc, char** argv) {
  if(argc!=2) return 2;
  std::ifstream input(argv[1]);
  if(!input) return 2;
  std::string line;std::size_t frames=0,current=0,held=0;
  while(std::getline(input,line)) {
    auto value=ParseIkVisualization(nlohmann::json::parse(line));
    if(!value.tianyi || value.displayed_ik().size()!=2) return 1;
    if(value.current_ik()) ++current; else ++held;
    // A network delay changes the label, not the preserved historical geometry.
    value.received-=std::chrono::seconds(2);
    if(value.current_ik() || value.displayed_ik().size()!=2) return 1;
    ++frames;
  }
  if(!current || !held) return 1;
  std::cout << "{\"frames\":" << frames << ",\"current\":" << current
            << ",\"held\":" << held << ",\"blank\":0,\"hardware_output\":false}\n";
}

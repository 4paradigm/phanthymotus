// Development-only: swift scripts/generate_operator_labels.swift > app/src/main/cpp/operator_labels.hpp
import AppKit
let labels=["开始遥操","结束并收臂","立即停止"]
print("#pragma once\n#include <array>\n#include <vector>\nnamespace motus::openxr_capture {\ninline const std::vector<std::array<int,3>> kOperatorLabelPixels = {")
for (n,label) in labels.enumerated() {
 let width=112, height=24
 let rep=NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:width,pixelsHigh:height,bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:width*4,bitsPerPixel:32)!
 NSGraphicsContext.saveGraphicsState();NSGraphicsContext.current=NSGraphicsContext(bitmapImageRep:rep)
 NSColor.black.setFill();NSRect(x:0,y:0,width:width,height:height).fill()
 (label as NSString).draw(at:NSPoint(x:0,y:0),withAttributes:[.font:NSFont.systemFont(ofSize:20),.foregroundColor:NSColor.white])
 NSGraphicsContext.restoreGraphicsState()
 for y in 0..<height {for x in 0..<width {
  if rep.colorAt(x:x,y:y)!.redComponent > 0.45 {print("{\(n),\(x),\(y)},")}
 }
}
}
print("};\n}")

# OCR Large Image Timeout Design

## Goal

Make the OCR plugin reliably process multi-megabyte compressed images on the
Jetson leaderboard without changing the MCP tool contract or output bbox
coordinate system.

## Root Causes

The camera subscriber uses best-effort delivery while the Fast DDS profile
forces large samples through UDP. A lost fragment therefore drops the complete
image. After delivery, the plugin fully decodes the source image, keeps up to
five compressed frames, rebuilds RapidOCR sessions for repeated empty config
calls, and removes stopped ROS nodes without destroying them. These behaviors
increase latency and allow memory use to accumulate until the process is killed.

## Design

Use reliable keep-last-one QoS for image and result topics. Keep the custom UDP
buffer profile, but do not disable Fast DDS built-in transports.

RapidOCR will read the source dimensions, decode large JPEG images at an
OpenCV-supported reduced resolution, resize other oversized formats after
decode, and scale result boxes back to source pixel coordinates. The configured
maximum side is 1600 pixels and remains configurable.

The plugin will retain one adapter for equivalent shared configuration. Empty
or language-only config calls do not recreate ONNX Runtime sessions. Inference
is serialized per adapter, frame queues keep only the newest image, and stopped
nodes are removed from the executor and destroyed. OCR leaderboard defaults use
one inference thread and disable ASR to avoid loading unrelated models.

## Compatibility

Inputs remain `sensor_msgs/CompressedImage`. Outputs retain `text`, `items`,
`timestamp`, `language`, and source-image pixel bboxes. The model files and
download URLs are unchanged.

## Verification

Unit tests cover QoS, reduced decode selection, bbox scaling, adapter reuse,
node destruction, queue bounds, Fast DDS packaging, and OCR-only resource
defaults. Existing OCR and repository tests must remain green.

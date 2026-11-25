/**
 * The one renderer list, in match order: the first renderer whose canRender()
 * accepts a topic's format draws it.
 *
 * The monitor page and the canvas detail panel used to keep a copy each, and
 * the copies drifted — sensor/pointcloud and then sensor/mapping drew on the
 * monitor page and showed a blank detail panel, because the detail copy never
 * got the new entry. Both import this list now; add a renderer here once.
 */

import { ActivityRenderer } from './activity.js';
import { TextRenderer }     from './text.js';
import { VideoRenderer }    from './video.js';
import { ImageRenderer }    from './image.js';
import { AudioRenderer }    from './audio.js';
import { LidarRenderer }    from './lidar.js';
import { ControlRenderer }  from './control.js';
import { PointCloudRenderer } from './pointcloud.js';
import { MappingRenderer }  from './mapping.js';
import { SkeletonRenderer } from './skeleton.js';
import { Pose2dRenderer }   from './pose2d.js';
import { TactileRenderer }  from './tactile.js';
import { BatteryRenderer }  from './battery.js';
import { KvLatestRenderer } from './kv-latest.js';
import { CameraRenderer, DepthRenderer, DepthZlibRenderer } from './camera.js';

export const RENDERERS = [VideoRenderer, CameraRenderer, DepthRenderer, DepthZlibRenderer, ImageRenderer, AudioRenderer, PointCloudRenderer, MappingRenderer, LidarRenderer, SkeletonRenderer, Pose2dRenderer, TactileRenderer, BatteryRenderer, ControlRenderer, TextRenderer, ActivityRenderer];

// Renderers a host selects by name rather than by canRender. KvLatest answers
// the same hints as Text — whoever sits earlier in the find list shadows the
// other in every view — so it stays out of RENDERERS and the monitor page's
// 日志/最新 toggle picks it explicitly.
export const EXPLICIT = { kv_latest: KvLatestRenderer };

/**
 * Renderers declare `channels`: the frame kinds they consume.
 *   'data'   topic frames over the WebSocket (onData(buffer))
 *   'event'  tool-call results / render events (onEvent(event))
 * The two carry different payload shapes, so a host that only delivers one
 * kind MUST filter by it. image/video/lidar are event-only: a topic card
 * with a video/depth hint used to select VideoRenderer in a data-only view
 * and starve — a blank card, no error anywhere. The channel filter falls
 * back to ActivityRenderer, which shows the no-data state instead.
 */
export function pickRenderer(hint, channel = null) {
  // channel=null → unfiltered: only for hosts that feed BOTH channels and
  // guard each feed with `?.` (the dashboard skill preview). Hosts that
  // deliver one frame kind MUST pass it.
  const capable = (r) => channel == null
    || (r.channels || ['data']).includes(channel);
  const Renderer = RENDERERS.find((r) => capable(r) && r.canRender(hint));
  return Renderer || ActivityRenderer;
}

export function instantiate(Renderer) {
  return Object.assign(Object.create(Object.getPrototypeOf(Renderer)), Renderer);
}

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
import { CameraRenderer, DepthRenderer, DepthZlibRenderer } from './camera.js';

export const RENDERERS = [VideoRenderer, CameraRenderer, DepthRenderer, DepthZlibRenderer, ImageRenderer, AudioRenderer, PointCloudRenderer, MappingRenderer, LidarRenderer, SkeletonRenderer, Pose2dRenderer, TactileRenderer, ControlRenderer, TextRenderer, ActivityRenderer];

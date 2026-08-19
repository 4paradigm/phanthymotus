from __future__ import annotations

from .contracts import ErrorCode, ObstacleDistanceError, SceneDomain


def _missing_scene(message: str) -> ObstacleDistanceError:
    return ObstacleDistanceError(ErrorCode.MISSING_SCENE, message)


def _coerce_scene(value: SceneDomain | str) -> SceneDomain:
    if isinstance(value, SceneDomain):
        return value
    if isinstance(value, str):
        try:
            return SceneDomain(value.strip().lower())
        except ValueError:
            pass
    raise _missing_scene("scene selection is invalid")


def resolve_scene(
    *,
    scene_hint: SceneDomain | str | None = None,
    fixed_scene: SceneDomain | str | None = None,
) -> SceneDomain:
    if scene_hint is not None:
        return _coerce_scene(scene_hint)

    if fixed_scene is not None:
        return _coerce_scene(fixed_scene)

    raise _missing_scene("scene selection is required")

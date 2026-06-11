import pathlib
import json
import fastapi

router = fastapi.APIRouter(prefix="/world")


@router.post("/world_save")
async def endpoint(
    world: dict = fastapi.Body(embed=True),
):
    id_ = world['id']
    path = pathlib.Path(f"./resource/world/{id_}.json")
    world_str = json.dumps(world, ensure_ascii=False, indent=4)
    path.write_text(world_str)

    return {
        'code': 200,
        'message': '',
        'data': {},
    }


@router.post("/world_load")
async def endpoint(
    id_: int = fastapi.Body(embed=True),
):
    path = pathlib.Path(f"./resource/world/{id_}.json")
    world = path.read_bytes()

    return world

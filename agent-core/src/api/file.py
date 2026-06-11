import os
import shutil
import aiofiles
import fastapi
import fastapi.responses


router = fastapi.APIRouter(prefix="/file")


# ========== list ==========
@router.post("/list")
async def endpoint(
    path: str = fastapi.Body('.', embed=True),
):
    entries = []
    for name in sorted(os.listdir(path)):
        entry_path = os.path.join(path, name)
        if os.path.isdir(entry_path):
            entries.append({'name': name, 'type': 'dir', 'size': 0})
        else:
            entries.append({'name': name, 'type': 'file', 'size': os.path.getsize(entry_path)})

    return {
        'code': 200,
        'message': '',
        'data': {
            'files': entries
        },
    }


# ========== read ==========
@router.post("/read")
async def endpoint(
    path: str = fastapi.Body(embed=True),
):
    try:
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            content = await f.read()
        encoding = 'utf-8'
    except UnicodeDecodeError:
        content = ''
        encoding = 'binary'

    return {
        'code': 200,
        'message': '',
        'data': {
            'content': content,
            'encoding': encoding,
        },
    }


# ========== write ==========
@router.post("/write")
async def endpoint(
    path: str = fastapi.Body(embed=True),
    content: str = fastapi.Body(embed=True),
):
    async with aiofiles.open(path, 'w', encoding='utf-8') as f:
        await f.write(content)

    return {
        'code': 200,
        'message': '',
        'data': {},
    }


# ========== upload ==========
@router.post("/upload")
async def endpoint(
    path: str = fastapi.Form('.'),
    file: fastapi.UploadFile = fastapi.File(),
):
    os.makedirs(path, exist_ok=True)
    file_full = os.path.join(path, file.filename)
    async with aiofiles.open(file_full, 'wb') as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)

    return {
        'code': 200,
        'message': '',
        'data': {},
    }


# ========== download ==========
@router.get("/download")
async def endpoint(
    path: str = fastapi.Query(),
):
    return fastapi.responses.FileResponse(path, filename=os.path.basename(path))


# ========== rename ==========
@router.post("/rename")
async def endpoint(
    path: str = fastapi.Body(embed=True),
    new_name: str = fastapi.Body(embed=True),
):
    new_path = os.path.join(os.path.dirname(path), new_name)
    os.rename(path, new_path)

    return {
        'code': 200,
        'message': '',
        'data': {},
    }


# ========== delete ==========
@router.post("/delete")
async def endpoint(
    path: str = fastapi.Body(embed=True),
):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

    return {
        'code': 200,
        'message': '',
        'data': {},
    }


if __name__ == "__main__":
    pass

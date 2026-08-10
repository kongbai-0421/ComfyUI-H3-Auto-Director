"""H3 Auto Director: sequential H3 generation with resumable AV state."""

import asyncio
import logging

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

try:
    from aiohttp import web
    from server import PromptServer
    from .upload import select_and_import_files, select_directory

    @PromptServer.instance.routes.post("/h3_auto_director/select_files")
    async def h3_auto_director_select_files(request):
        try:
            payload = await request.json()
            file_type = payload.get("type", "")
            initial_dir = payload.get("initial_dir", "")
            use_default_path = bool(payload.get("use_default_path", True))
            files = await asyncio.to_thread(select_and_import_files, file_type, initial_dir, use_default_path)
            return web.json_response({"files": files})
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            logging.getLogger("h3_auto_director").exception("Native reference picker failed")
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.post("/h3_auto_director/select_directory")
    async def h3_auto_director_select_directory(request):
        try:
            payload = await request.json()
            directory = await asyncio.to_thread(select_directory, payload.get("initial_dir", ""))
            return web.json_response({"directory": directory})
        except Exception as exc:
            logging.getLogger("h3_auto_director").exception("Native directory picker failed")
            return web.json_response({"error": str(exc)}, status=500)
except Exception as exc:
    logging.getLogger("h3_auto_director").warning("Native reference picker unavailable: %s", exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

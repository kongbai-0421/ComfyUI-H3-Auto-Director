"""H3 Auto Director: sequential H3 generation with resumable AV state."""

import asyncio
import logging

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

try:
    from aiohttp import web
    from server import PromptServer
    from .upload import select_and_import_files

    @PromptServer.instance.routes.post("/h3_auto_director/select_files")
    async def h3_auto_director_select_files(request):
        try:
            payload = await request.json()
            file_type = payload.get("type", "")
            initial_dir = payload.get("initial_dir", "")
            files = await asyncio.to_thread(select_and_import_files, file_type, initial_dir)
            return web.json_response({"files": files})
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            logging.getLogger("h3_auto_director").exception("Native reference picker failed")
            return web.json_response({"error": str(exc)}, status=500)
except Exception as exc:
    logging.getLogger("h3_auto_director").warning("Native reference picker unavailable: %s", exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

import asyncio

import log


class Event():
    def __init__(self):
        pass

    
    @log.function_(call=True)
    async def __call__(self):
        """结束当前任务，表示所有步骤已执行完毕"""

        return '完成'

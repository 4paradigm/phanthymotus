import functools
import traceback


import warnings
warnings.filterwarnings("ignore")


import collections
import itertools
import logging
import os

logger = logging.getLogger("main")
# Was setLevel(1) — below DEBUG(10), i.e. "emit absolutely everything forever".
# That is a production log-volume amplifier that never shows up in a grep for
# logging.DEBUG. Default to INFO, overridable for debugging.
logger.setLevel(getattr(logging, os.environ.get('AGENT_CORE_LOG_LEVEL', 'INFO').upper(), logging.INFO))

# The in-memory buffer behind /api/logging/*_stream. Was an unbounded list, so a
# long-lived core container grew it forever.
LOG_BUFFER_SIZE = int(os.environ.get('AGENT_CORE_LOG_BUFFER', '2000'))


class LoggingHandler(logging.Handler):
    def __init__(self, maxlen: int = None):
        super().__init__()
        self.record_list = collections.deque(maxlen=maxlen or LOG_BUFFER_SIZE)
        self.dropped = 0
        self._seq = itertools.count(1)

    def emit(self, record):
        # Called under self.lock by Handler.handle(), so no extra locking needed.
        # Stamp a monotonic seq: consumers cannot use list indices any more,
        # because a full deque stops growing and len() would stall them.
        record.seq = next(self._seq)
        if len(self.record_list) == self.record_list.maxlen:
            self.dropped += 1
        print(self.format(record))
        self.record_list.append(record)

handler = LoggingHandler()
logger.addHandler(handler)


def function_(
    call: bool = False,
    input: bool = False, 
    exception: bool = False, 
    exception_detail: bool = False,
    exception_raise: bool = True,
    output: bool = False, 
):
    def decorator(function_):
        @functools.wraps(function_)
        async def wrapper(*args, **kwargs):
            function_path = function_.__module__ + '.' + function_.__qualname__

            try:
                if call: logger.info('<%s>[调用]', function_path)
                if input: logger.info('<%s>[输入][%s][%s]', function_path, args, kwargs)
                result = await function_(*args, **kwargs)
                if output: logger.info('<%s>[输出][%s]', function_path, result)
                return result
            
            except Exception as e:
                e_full = traceback.format_exc()
                if exception: logger.error('<%s>[发生错误][%s][%s]', function_path, e, e_full)
                if exception_raise: raise e

            result = await function_(*args, **kwargs)
            return result

        return wrapper
    return decorator


# def _log_async(*, level):
#     def decorator(function_):
#         function_signature = inspect.signature(function_)

#         @functools.wraps(function_)
#         async def wrapper(*args, **kwargs):
#             function_path = function_.__module__ + '.' + function_.__qualname__ + '()'
#             funcion_id = yuid()

#             function_args = function_signature.bind(*args, **kwargs)
#             function_args.arguments.pop('self', None)
#             function_args.arguments.pop('cls', None)
#             function_args = dict(function_args.arguments)

#             try:
#                 logger.log(level, '', extra = {
#                     'funciton':{
#                         'path': function_path,
#                         'status': 'before',
#                         'id': funcion_id,
                        
#                         'time_start': time.time(),
#                         'args': function_args
#                     },
#                 })

#                 time_start = time.monotonic()
#                 result = await function_(*args, **kwargs)
    
#                 logger.log(level, '', extra = {
#                     'funciton':{
#                         'path': function_path,
#                         'status': 'before',
#                         'id': funcion_id,

#                         'time_elapsed': time.monotonic() - time_start,
#                         'result': result
#                     },
#                 })
#                 return result
            
#             except Exception as e:
#                 logger.log(level, '', extra = {
#                     'funciton':{
#                         'path': function_path,
#                         'status': 'before',
#                         'id': funcion_id,

#                         'time_elapsed': time.monotonic() - time_start,
#                         'exception': e
#                     },
#                 })
#                 raise

#         return wrapper
#     return decorator






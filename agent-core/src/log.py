import functools
import traceback


import warnings
warnings.filterwarnings("ignore")


import logging
logger = logging.getLogger("main")
logger.setLevel(1)

class LoggingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.record_list = []

    def emit(self, record):
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






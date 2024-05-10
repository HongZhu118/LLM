import traceback

def exception_decorator(f):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            stack_trace = traceback.format_exc()
            logger.error(stack_trace)
            app.publish(
                "ErrMsg",
                {"context": g.context_queue.get(), "err_msg": stack_trace})

    return wrapper

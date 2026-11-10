"""Minimal vendored backport of exceptiongroup for offline environments.

This is intentionally tiny: FastAPI's anyio only uses `BaseExceptionGroup`
and `ExceptionGroup` for raising grouped exceptions. We avoid pulling in
the heavier optional bits (format, split, subgroup, derive) unless a caller
asks for them, to keep startup fast and the dependency surface zero.
"""

__all__ = [
    "BaseExceptionGroup",
    "ExceptionGroup",
    "catch",
]

# Try native first, then fall back to our vendored version
try:
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup, catch
except ImportError:
    try:
        import sys
        sys.path.insert(0, str(__file__).rpartition('/')[0])
        from exceptiongroup import BaseExceptionGroup, ExceptionGroup, catch
    except ImportError:
        class BaseExceptionGroup(BaseException):
            def __new__(cls, message, exceptions):
                if not exceptions:
                    raise ValueError("second argument must be a non-empty tuple")
                if not isinstance(exceptions, tuple):
                    raise TypeError(f"second argument must be a tuple, not {type(exceptions).__name__}")
                if not all(isinstance(e, BaseException) for e in exceptions):
                    raise TypeError("all items in the exceptions sequence must be BaseException instances")

                instance = super().__new__(cls, message, exceptions)
                instance.exceptions = exceptions
                instance.__cause__ = None
                instance.__context__ = None
                return instance

            def __str__(self):
                return f"{self.args[0]} ({', '.join(repr(e) for e in self.exceptions)})"

            @classmethod
            def _from_nested(cls, message, excs):
                return cls(message, excs)

        class ExceptionGroup(BaseExceptionGroup, Exception):
            @classmethod
            def _from_nested(cls, message, excs):
                return cls(message, excs)

        def catch(handlers, *, resource_warnings=False):
            import contextlib
            @contextlib.contextmanager
            def cm():
                try:
                    yield
                except BaseExceptionGroup as eg:
                    match = tuple(e for e in eg.exceptions if _is_match(e, handlers))
                    rest = tuple(e for e in eg.exceptions if not _is_match(e, handlers))
                    if match:
                        _dispatch(match, handlers, resource_warnings)
                    if rest:
                        raise ExceptionGroup(str(eg), rest)
            return cm()

        def _is_match(exc, handlers):
            for exc_type, handler in handlers.items():
                if isinstance(exc, exc_type):
                    return True
            return False

        def _dispatch(excs, handlers, resource_warnings):
            for exc_type, handler in handlers.items():
                for exc in excs:
                    if isinstance(exc, exc_type):
                        try:
                            handler(ExceptionGroup(str(exc), (exc,)))
                        except Exception:
                            import traceback
                            traceback.print_exc()
                        else:
                            break

        class ExceptionGroup(BaseExceptionGroup, Exception):
            @classmethod
            def _from_nested(cls, message, excs):
                return cls(message, excs)


try:
    import sys
    sys.path.insert(0, str(__file__).rpartition('/')[0])
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup, catch
except ImportError:
    # Use vendored versions
    BaseExceptionGroup = BaseExceptionGroup
    ExceptionGroup = ExceptionGroup
    catch = catch

__all__.append('BaseExceptionGroup')
__all__.append('ExceptionGroup')
__all__.append('catch')

import ezsim


_seen: set[str] = set()


def warn_once(message: str):
    global _seen
    if message in _seen:
        return
    _seen.add(message)
    ezsim.logger.warning(message)

class BaseSurface:
    _surfaces = dict()

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            BaseSurface._surfaces[name.lower()] = cls
            cls._name = name
        else:
            BaseSurface._surfaces[cls.__name__.lower()] = cls
            cls._name = cls.__name__


def get_surface(surface_config: dict):
    name = surface_config.pop("name").lower()
    return BaseSurface._surfaces[name](**surface_config)

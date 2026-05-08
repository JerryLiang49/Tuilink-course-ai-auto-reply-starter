from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    """Shared base class for all project data models.

    The workflow passes Pydantic model instances between nodes, and the prompts
    often include those instances directly as strings. Returning ``repr`` from
    ``__str__`` keeps string rendering explicit and field-oriented, which makes
    prompt inputs easier to inspect while debugging.
    """

    def __str__(self):
        return self.__repr__()

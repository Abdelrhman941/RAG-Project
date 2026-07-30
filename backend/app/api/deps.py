from typing import Annotated

from fastapi import Depends

from ..core import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]

from pydantic import BaseModel
from typing import List

class Item(BaseModel):
    text: List[str]

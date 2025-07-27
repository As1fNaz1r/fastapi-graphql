from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional


fake_items =[
    {"name":"pen",
    "price":10.1},
    {"name":"pencil",
    "price": 12.3}
]

app = FastAPI()

class Item(BaseModel):
    name:str = Field(..., min_length=2, max_length=50, example='Apple')
    price:float = Field(..., gt=0, example=10.33)
    description: Optional[str] = Field(None, max_length=200, example='Optional Description')


class ItemWithID(Item):
    id:int

# In-memory DB
fake_db:List[ItemWithID] = []
next_id = 0

@app.get("/")
def read_root():
    return {"message":"hello root!"}


@app.post("/items/", response_model=ItemWithID, status_code=status.HTTP_201_CREATED)
def create_item(item:Item):
    global next_id
    next_item = ItemWithID(**item.dict(), id=next_id)
    fake_db.append(next_item)
    next_id += 1
    return next_item


@app.get("/items")
def get_items(limit: int = 10):
    return {"message": f"returning {limit} items"}


@app.get("/items/{item_id}", response_model=ItemWithID)
def get_item(item_id:int):
    for item in fake_db:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="item not found")
    


@app.put("/items/{item_id}", response_model=ItemWithID)
def update_item(item_id:int, item:Item):
    for i, old_item in enumerate(fake_db):
        if old_item.id == item_id:
            updated = ItemWithID(**item.dict(), id=item_id)
            fake_db[i]=updated
            return updated
    raise HTTPException(status_code=404,detail='Item not found')


@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id:int):
    for i, item in enumerate(fake_db):
        if item.id == item_id:
            fake_db.pip(i)
            return
    raise HTTPException(status_code=404,detail="Item not found")

@app.get("/all-items", response_model=List[ItemWithID])
def read_all_items():
    return fake_db



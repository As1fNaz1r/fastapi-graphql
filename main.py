from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from typing import List, Optional


fake_items =[
    {"name":"pen",
    "price":10.1},
    {"name":"pencil",
    "price": 12.3}
]

app = FastAPI()

class Item(BaseModel):
    name:str
    price:float


class ItemWithID(Item):
    id:int

# In-memory DB
fake_db:List[ItemWithID] = []
next_id = 0

@app.get("/")
def read_root():
    return {"message":"hello root!"}


@app.get("/hello/{name}")
def read_name(name:str):
    return {'message':f'hello my name is {name}'}


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


@app.put("/items/{item_id}")
def update_item(item_id:int, item:Item):
    return{"item_id": item_id, "updated":item}


@app.delete("/items/{item_id}")
def delete_item(item_id:int):
    return {"message":f"item {item_id} deleted"}

@app.get("/all-items", response_model=List[ItemWithID])
def read_all_items():
    return fake_db



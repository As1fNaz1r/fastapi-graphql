from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


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

@app.get("/")
def read_root():
    return {"message":"hello root!"}


@app.get("/hello/{name}")
def read_name(name:str):
    return {'message':f'hello my name is {name}'}


@app.post("/items/")
def create_item(item:Item):
    return {"name": item.name, "price": item.price}


@app.get("/items")
def get_items(limit: int = 10):
    return {"message": f"returning {limit} items"}


@app.put("/items/{item_id}")
def update_item(item_id:int, item:Item):
    return{"item_id": item_id, "updated":item}


@app.delete("/items/{item_id}")
def delete_item(item_id:int):
    return {"message":f"item {item_id} deleted"}

@app.get("/all-items", response_model=List[Item])
def read_all_items():
    return fake_items



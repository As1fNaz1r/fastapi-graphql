from fastapi import FastAPI
from pydantic import BaseModel


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
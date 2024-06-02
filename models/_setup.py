from sqlmodel import SQLModel

def setup_database(engine):
    SQLModel.metadata.create_all(engine)
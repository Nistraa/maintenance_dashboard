from sqlmodel import SQLModel

def setup_database(engine):
    SQLModel.metadata.create_all(bind=engine)

def teardown_database(engine):
    SQLModel.metadata.drop_all(bind=engine)
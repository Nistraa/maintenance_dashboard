from sqlmodel import create_engine, Session, StaticPool


DATABASE_ENGINE = create_engine('sqlite:///db.sqlite')

'''
database engine for testin purpose
StaticPool allows to test the same database over several tests
'''
DATABASE_TESTING_ENGINE = create_engine(
    'sqlite:///:memory:',
    connect_args={
        "check_same_thread": False,
    },
   poolclass=StaticPool
)

def get_db():
    with Session(
        bind=DATABASE_ENGINE,
        autocommit=False,
        autoflush=False
    ) as db:
        try:
            yield db
        finally:
            db.close()

def get_testing_db():
    with Session(
        bind=DATABASE_TESTING_ENGINE,
        autocommit=False,
        autoflush=False,
    ) as db:
        try:
            yield db
        finally:
            db.close()

from sqlmodel import create_engine, Session


DATABASE_ENGINE = create_engine('sqlite:///db.sqlite')
session_local = Session(bind=DATABASE_ENGINE, autocommit=False, autoflush=False)


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

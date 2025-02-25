from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Float
from web.database.base_model import AbstractBase
from web.media_handler.schemas import MediaSchema


class Media(AbstractBase):
    __tablename__ = "media"
    __pydantic_model__ = MediaSchema

    url: Mapped[str] = mapped_column(String(500), nullable=True)
    car_type_body: Mapped[str] = mapped_column(String(500), nullable=True)
    car_type_body_score: Mapped[str] = mapped_column(Float(), nullable=True)

    def to_read_model(self):
        return self.__pydantic_model__(
            id=self.id,
            url=self.url,
            result=self.result,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


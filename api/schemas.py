from pydantic import BaseModel, Field, ConfigDict


class BookRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True
    )

    book_title: str = Field(
        alias="Book-Title",
        min_length=1
    )

    book_author: str = Field(
        alias="Book-Author",
        min_length=1
    )

    year: float = Field(
        alias="Year-Of-Publication",
        ge=1000,
        le=2026
    )

    publisher: str = Field(
        alias="Publisher",
        min_length=1
    )

"""Request / response models for the HTTP API (validated by pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src import config

Category = Literal["yeni tikili", "kohne tikili", "heyet evi/bag evi"]


class ListingIn(BaseModel):
    """A listing as a person describes it. Leave a field out if you don't know it."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "category": "yeni tikili", "area_m2": 90, "rooms": 3, "floor": 7, "total_floors": 16,
        "location": "28 may m.", "repair": True, "bill_of_sale": True}})

    category: Category = Field(description="New building, old building, or house / villa")
    area_m2: float = Field(ge=config.AREA_RANGE_M2[0], le=config.AREA_RANGE_M2[1])
    rooms: int | None = Field(None, ge=config.ROOMS_RANGE[0], le=config.ROOMS_RANGE[1])
    floor: int | None = Field(None, ge=-2, le=config.MAX_TOTAL_FLOORS)
    total_floors: int | None = Field(None, ge=1, le=config.MAX_TOTAL_FLOORS)
    land_area_sot: float | None = Field(None, ge=0, le=10_000, description="Houses only; 1 sot = 100 m²")
    location: str | None = Field(None, max_length=80, description="A value from /api/options")
    lat: float | None = Field(None, ge=config.LAT_RANGE[0], le=config.LAT_RANGE[1])
    lng: float | None = Field(None, ge=config.LNG_RANGE[0], le=config.LNG_RANGE[1])
    repair: bool | None = Field(None, description="Renovated?")
    mortgage: bool = False
    bill_of_sale: bool = Field(False, description="Has a 'kupça' (bill of sale)")
    description: str = Field("", max_length=5_000)

    @model_validator(mode="after")
    def _consistent(self):
        if self.floor is not None and self.total_floors is not None and self.floor > self.total_floors:
            raise ValueError("floor cannot be above the number of floors in the building")
        if (self.lat is None) != (self.lng is None):
            raise ValueError("give both lat and lng, or neither")
        return self


class LocationUsed(BaseModel):
    location: str | None
    district: str | None
    city: str | None
    lat: float | None
    lng: float | None
    coordinates_from: str


class Factor(BaseModel):
    factor: str
    log_effect: float
    pct_effect: float = Field(description="Effect on the price in %, relative to the typical listing")


class PredictionOut(BaseModel):
    price_azn: float
    price_low_azn: float = Field(description="Lower end of the 80% range (split-conformal on out-of-bag residuals)")
    price_high_azn: float
    price_per_m2_azn: float
    premium: bool
    tier_threshold_azn: float = Field(description="Premium means a price above this (train median)")
    svm_score: float
    inside_margin: bool = Field(description="|score| < 1: the SVM is less sure here")
    models_agree: bool = Field(description="Does the price estimate fall on the same side of the threshold?")
    trees_low_azn: float = Field(description="10th percentile of the forest's individual trees")
    trees_high_azn: float = Field(description="90th percentile of the forest's individual trees")
    factors: list[Factor] = Field(description="Largest drivers of this estimate (forest path decomposition)")
    location_used: LocationUsed
    warnings: list[str]

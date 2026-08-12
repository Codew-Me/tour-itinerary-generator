"""SQLAlchemy models."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class District(Base):
    __tablename__ = "districts"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    normalized_name = Column(String(100), unique=True, nullable=False)

    destinations = relationship("Destination", back_populates="district")


class Destination(Base):
    __tablename__ = "destinations"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    normalized_name = Column(String(200), nullable=False)
    district_id = Column(Integer, ForeignKey("districts.id"), nullable=False)

    district = relationship("District", back_populates="destinations")
    attractions = relationship("Attraction", back_populates="destination")
    review_links = relationship("ReviewDestinationLink", back_populates="destination")

    __table_args__ = (UniqueConstraint("name", "district_id", name="uq_destination_district"),)


class Attraction(Base):
    __tablename__ = "attractions"

    id = Column(Integer, primary_key=True)
    name = Column(String(300), unique=True, nullable=False)
    normalized_name = Column(String(300), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    mood = Column(String(50), nullable=False, index=True)
    details = Column(Text, nullable=False)
    image_url = Column(Text, nullable=True)
    destination_id = Column(Integer, ForeignKey("destinations.id"), nullable=False)
    review_count = Column(Integer, default=0, nullable=False)
    review_evidence_available = Column(Boolean, default=False, nullable=False)
    match_type = Column(String(30), default="none", nullable=False)

    destination = relationship("Destination", back_populates="attractions")
    review_links = relationship("AttractionReviewLink", back_populates="attraction")
    saved_by = relationship("SavedDestination", back_populates="attraction")


class AttractionReviewLink(Base):
    __tablename__ = "attraction_review_links"

    id = Column(Integer, primary_key=True)
    attraction_id = Column(Integer, ForeignKey("attractions.id"), nullable=False)
    review_destination_name = Column(String(300), nullable=False)

    attraction = relationship("Attraction", back_populates="review_links")

    __table_args__ = (
        UniqueConstraint("attraction_id", "review_destination_name", name="uq_attraction_review_dest"),
    )


class ReviewDestinationLink(Base):
    """Review-only destinations not tied to a single attraction."""

    __tablename__ = "review_destination_links"

    id = Column(Integer, primary_key=True)
    review_destination_name = Column(String(300), unique=True, nullable=False)
    destination_id = Column(Integer, ForeignKey("destinations.id"), nullable=True)
    review_count = Column(Integer, default=0, nullable=False)

    destination = relationship("Destination", back_populates="review_links")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    saved_destinations = relationship("SavedDestination", back_populates="user")
    conversations = relationship("ConversationSession", back_populates="user")


class SavedDestination(Base):
    __tablename__ = "saved_destinations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    attraction_id = Column(Integer, ForeignKey("attractions.id"), nullable=False)
    saved_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="saved_destinations")
    attraction = relationship("Attraction", back_populates="saved_by")

    __table_args__ = (UniqueConstraint("user_id", "attraction_id", name="uq_user_attraction_save"),)


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(300), nullable=False, default="New conversation")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversation_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user | assistant | tool
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversation = relationship("ConversationSession", back_populates="messages")


class RecommendationScore(Base):
    """Cache explainable scores for destinations."""

    __tablename__ = "recommendation_scores"

    id = Column(Integer, primary_key=True)
    query_hash = Column(String(64), nullable=False, index=True)
    attraction_id = Column(Integer, ForeignKey("attractions.id"), nullable=False)
    preference_match = Column(Float, default=0.0)
    review_relevance = Column(Float, default=0.0)
    category_match = Column(Float, default=0.0)
    mood_match = Column(Float, default=0.0)
    location_match = Column(Float, default=0.0)
    evidence_availability = Column(Float, default=0.0)
    total_score = Column(Float, default=0.0)

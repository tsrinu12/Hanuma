"""SQLAlchemy models for the gamification domain."""
from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class VideoStats(Base):
    __tablename__ = "video_stats"
    video_id = Column(String(64), primary_key=True)
    views = Column(BigInteger, default=0, nullable=False)
    unique_viewers = Column(BigInteger, default=0, nullable=False)
    completes = Column(BigInteger, default=0, nullable=False)
    likes = Column(BigInteger, default=0, nullable=False)
    comments = Column(BigInteger, default=0, nullable=False)
    shares = Column(BigInteger, default=0, nullable=False)
    watch_seconds = Column(BigInteger, default=0, nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class UserPoints(Base):
    __tablename__ = "user_points"
    user_id = Column(String(64), primary_key=True)
    points = Column(BigInteger, default=0, nullable=False)
    level = Column(Integer, default=1, nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class PointEvent(Base):
    __tablename__ = "point_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), index=True, nullable=False)
    video_id = Column(String(64), index=True, nullable=True)
    action = Column(String(32), nullable=False)
    points = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class UserStreak(Base):
    __tablename__ = "user_streaks"
    user_id = Column(String(64), primary_key=True)
    current_streak = Column(Integer, default=0, nullable=False)
    longest_streak = Column(Integer, default=0, nullable=False)
    last_active_day = Column(Date, nullable=True)


class Badge(Base):
    __tablename__ = "badges"
    code = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    description = Column(String(512), nullable=False)
    points_required = Column(Integer, default=0)


class UserBadge(Base):
    __tablename__ = "user_badges"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), index=True, nullable=False)
    badge_code = Column(String(64), ForeignKey("badges.code"), nullable=False)
    awarded_at = Column(DateTime, default=func.now())
    __table_args__ = (UniqueConstraint("user_id", "badge_code", name="uq_user_badge"),)


class WatchSession(Base):
    __tablename__ = "watch_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), index=True, nullable=False)
    video_id = Column(String(64), index=True, nullable=False)
    fraction_watched = Column(Integer, default=0)
    counted_as_view = Column(Integer, default=0)
    counted_as_complete = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())


def init_db() -> None:
    Base.metadata.create_all(bind=engine)

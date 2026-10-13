# feature_repo/features.py
"""
Feast feature definitions for IT Incident Routing.

Defines:
- Entities: Primary join key (incident_id, category)
- Data Sources: Parquet offline source
- Feature Views: Aggregated category / priority routing features
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

# =============================================================================
# ENTITIES
# =============================================================================
# Primary keys used to look up historical and contextual incident features
category_entity = Entity(
    name="category",
    description="IT Service Category (e.g., Software, Hardware, Network, Access)",
    value_type=ValueType.STRING,
)

# =============================================================================
# DATA SOURCES
# =============================================================================
category_stats_source = FileSource(
    name="category_routing_stats_source",
    path="../data/category_routing_stats.parquet",
    timestamp_field="event_timestamp",
)

# =============================================================================
# FEATURE VIEWS
# =============================================================================
category_routing_fv = FeatureView(
    name="category_routing_stats",
    description="Aggregated historical routing distributions and volume by IT category",
    entities=[category_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="historical_volume", dtype=Int64, description="Monthly incident count in category"),
        Field(name="reassignment_ratio", dtype=Float32, description="Historical reassignment rate"),
        Field(name="top_resolver_group", dtype=String, description="Most frequent resolver group"),
    ],
    source=category_stats_source,
    online=True,
)

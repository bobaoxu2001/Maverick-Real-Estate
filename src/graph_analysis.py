"""
Graph / Network Analysis of NYC CRE Market Participants
========================================================
Models the network of relationships between property owners, buildings,
and market participants using graph-based techniques.

Mirrors Maverick's Neo4j graph database approach, capturing:
  - Owner → Property relationships
  - Co-ownership networks (shared LLCs, management companies)
  - Geographic proximity clusters
  - Transaction flow networks (buyer → seller chains)

This module provides both:
  1. NetworkX implementation (works locally, no Neo4j required)
  2. Neo4j Cypher queries (production-ready for graph database)

Author: Allen Xu
"""

import numpy as np
import pandas as pd
import networkx as nx
from loguru import logger
from collections import Counter

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


# ─────────────────────────────────────────────────────────
# NetworkX Implementation (Local)
# ─────────────────────────────────────────────────────────
class CRENetworkAnalyzer:
    """Graph analysis of NYC CRE market participant networks.

    Builds and analyzes a bipartite graph of owners and properties,
    identifying influential players, portfolio clusters, and
    hidden connections in the market.
    """

    def __init__(self):
        self.G = nx.Graph()
        self.owner_graph = None  # Projected owner-to-owner graph
        self.results = {}

    def build_ownership_graph(self, property_df: pd.DataFrame) -> nx.Graph:
        """Build bipartite owner-property graph from PLUTO/sales data.

        Nodes:
          - Owner nodes (type='owner')
          - Property nodes (type='property')
        Edges:
          - Owner → Property (ownership relationship)
        """
        self.G = nx.Graph()

        owner_col = None
        for col in ["ownername", "owner_name", "owner"]:
            if col in property_df.columns:
                owner_col = col
                break

        if owner_col is None:
            logger.warning("No owner column found in data")
            return self.G

        bbl_col = "bbl" if "bbl" in property_df.columns else None

        for _, row in property_df.iterrows():
            owner = str(row.get(owner_col, "")).strip().upper()
            if not owner or owner == "NAN":
                continue

            property_id = str(row.get(bbl_col, "")) if bbl_col else str(row.name)

            # Add owner node
            if not self.G.has_node(f"owner:{owner}"):
                self.G.add_node(
                    f"owner:{owner}",
                    node_type="owner",
                    name=owner,
                )

            # Add property node
            prop_attrs = {"node_type": "property", "bbl": property_id}
            for attr in ["borough_name", "neighborhood", "assesstot", "bldgarea", "latitude", "longitude"]:
                if attr in row.index:
                    prop_attrs[attr] = row[attr]

            self.G.add_node(f"property:{property_id}", **prop_attrs)

            # Add ownership edge
            edge_attrs = {}
            if "sale_price" in row.index:
                edge_attrs["sale_price"] = row["sale_price"]
            if "sale_date" in row.index:
                edge_attrs["sale_date"] = str(row["sale_date"])

            self.G.add_edge(f"owner:{owner}", f"property:{property_id}", **edge_attrs)

        n_owners = sum(1 for n, d in self.G.nodes(data=True) if d.get("node_type") == "owner")
        n_props = sum(1 for n, d in self.G.nodes(data=True) if d.get("node_type") == "property")

        logger.info(
            f"Ownership graph: {n_owners} owners, {n_props} properties, "
            f"{self.G.number_of_edges()} edges"
        )
        return self.G

    def project_owner_network(self) -> nx.Graph:
        """Project bipartite graph to owner-owner network.

        Two owners are connected if they own properties in the
        same neighborhood, building, or are linked through
        transaction chains. Edge weight = number of shared contexts.

        This reveals hidden connections between market participants.
        """
        # Get owner and property nodes
        owners = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "owner"]
        properties = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "property"]

        # Create owner-to-owner projection
        self.owner_graph = nx.Graph()

        # Connect owners who own properties in the same neighborhood
        neighborhood_groups = {}
        for prop in properties:
            hood = self.G.nodes[prop].get("neighborhood", "Unknown")
            if hood not in neighborhood_groups:
                neighborhood_groups[hood] = []
            owners_of_prop = [n for n in self.G.neighbors(prop) if n.startswith("owner:")]
            neighborhood_groups[hood].extend(owners_of_prop)

        for hood, hood_owners in neighborhood_groups.items():
            unique_owners = list(set(hood_owners))
            for i in range(len(unique_owners)):
                for j in range(i + 1, len(unique_owners)):
                    if self.owner_graph.has_edge(unique_owners[i], unique_owners[j]):
                        self.owner_graph[unique_owners[i]][unique_owners[j]]["weight"] += 1
                        self.owner_graph[unique_owners[i]][unique_owners[j]]["shared_neighborhoods"].append(hood)
                    else:
                        self.owner_graph.add_edge(
                            unique_owners[i], unique_owners[j],
                            weight=1,
                            shared_neighborhoods=[hood],
                        )

        logger.info(
            f"Owner network: {self.owner_graph.number_of_nodes()} owners, "
            f"{self.owner_graph.number_of_edges()} connections"
        )
        return self.owner_graph

    def identify_key_players(self, top_n: int = 20) -> pd.DataFrame:
        """Identify the most influential market participants using centrality metrics.

        Metrics:
          - Degree centrality: Number of properties owned
          - Betweenness centrality: Bridge between market clusters
          - Eigenvector centrality: Connected to other influential owners
          - Portfolio value: Total assessed value of owned properties
        """
        if self.G.number_of_nodes() == 0:
            logger.warning("Empty graph — build graph first")
            return pd.DataFrame()

        owners = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "owner"]

        metrics = []
        for owner in owners:
            properties_owned = [
                n for n in self.G.neighbors(owner)
                if self.G.nodes[n].get("node_type") == "property"
            ]

            # Portfolio value
            portfolio_value = sum(
                float(self.G.nodes[p].get("assesstot", 0) or 0)
                for p in properties_owned
            )

            # Neighborhoods
            neighborhoods = set(
                self.G.nodes[p].get("neighborhood", "Unknown")
                for p in properties_owned
            )

            metrics.append({
                "owner": self.G.nodes[owner].get("name", owner),
                "n_properties": len(properties_owned),
                "portfolio_value": portfolio_value,
                "n_neighborhoods": len(neighborhoods),
                "neighborhoods": list(neighborhoods)[:5],
                "degree_centrality": nx.degree_centrality(self.G).get(owner, 0),
            })

        df = pd.DataFrame(metrics)
        df = df.sort_values("portfolio_value", ascending=False).head(top_n)

        # Add network centrality from owner graph if available
        if self.owner_graph and self.owner_graph.number_of_nodes() > 0:
            try:
                betweenness = nx.betweenness_centrality(self.owner_graph)
                df["betweenness_centrality"] = df["owner"].apply(
                    lambda x: betweenness.get(f"owner:{x}", 0)
                )
            except Exception:
                pass

        self.results["key_players"] = df
        logger.info(f"Top {top_n} market participants identified")
        return df

    def detect_communities(self) -> dict:
        """Detect market communities using graph algorithms.

        Uses Louvain community detection to find clusters of
        owners and properties that form natural market segments.
        """
        if self.G.number_of_nodes() == 0:
            return {}

        try:
            communities = nx.community.louvain_communities(self.G, seed=42)
            community_info = []
            for i, community in enumerate(communities):
                owners_in = [n for n in community if n.startswith("owner:")]
                props_in = [n for n in community if n.startswith("property:")]
                community_info.append({
                    "community_id": i,
                    "n_owners": len(owners_in),
                    "n_properties": len(props_in),
                    "total_nodes": len(community),
                })

            self.results["communities"] = community_info
            logger.info(f"Detected {len(communities)} communities")
            return {"n_communities": len(communities), "details": community_info}

        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {}

    def find_propensity_to_sell_signals(self, property_df: pd.DataFrame) -> pd.DataFrame:
        """Identify properties with elevated propensity to sell.

        Signals from network analysis:
          1. Owner has been selling other properties recently
          2. Owner's portfolio is concentrated (single-asset risk)
          3. Properties in neighborhoods with high turnover
          4. Owner connected to distressed owners in network
        """
        df = property_df.copy()
        sell_scores = pd.Series(0.0, index=df.index)

        # Signal 1: Owner selling pattern
        owner_col = None
        for col in ["ownername", "owner_name", "owner"]:
            if col in df.columns:
                owner_col = col
                break

        if owner_col:
            owner_counts = df[owner_col].value_counts()
            # Owners with only 1 property are more likely to sell (less committed)
            single_asset_owners = set(owner_counts[owner_counts == 1].index)
            sell_scores += df[owner_col].isin(single_asset_owners).astype(float) * 0.3

        # Signal 2: High turnover neighborhood
        if "neighborhood" in df.columns and "sale_date" in df.columns:
            df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
            recent_mask = df["sale_date"] >= (pd.Timestamp.now() - pd.Timedelta(days=365*2))
            turnover = df[recent_mask].groupby("neighborhood").size()
            high_turnover = set(turnover[turnover > turnover.quantile(0.75)].index)
            sell_scores += df["neighborhood"].isin(high_turnover).astype(float) * 0.2

        # Signal 3: Below-market value (motivated seller)
        if "price_vs_neighborhood" in df.columns:
            pvn = pd.to_numeric(df["price_vs_neighborhood"], errors="coerce")
            sell_scores += (pvn < 0.8).astype(float) * 0.25

        # Signal 4: Aging building without renovation
        if "building_age" in df.columns and "is_recently_renovated" in df.columns:
            aged = (pd.to_numeric(df["building_age"], errors="coerce") > 50) & (df["is_recently_renovated"] == 0)
            sell_scores += aged.astype(float) * 0.25

        df["propensity_to_sell_score"] = sell_scores.clip(0, 1)
        df["sell_likelihood"] = pd.cut(
            df["propensity_to_sell_score"],
            bins=[0, 0.2, 0.4, 0.6, 1.0],
            labels=["Low", "Moderate", "Elevated", "High"],
        )

        logger.info(f"Sell propensity scores: {df['sell_likelihood'].value_counts().to_dict()}")
        return df


# ─────────────────────────────────────────────────────────
# Neo4j Cypher Queries (Production Reference)
# ─────────────────────────────────────────────────────────
NEO4J_SCHEMA = """
// ─── Neo4j Graph Schema for NYC CRE ───
// Node Labels:
//   (:Owner {name, type, portfolio_value})
//   (:Property {bbl, address, borough, neighborhood, assessed_value, lat, lon})
//   (:Neighborhood {name, borough, avg_price})
//   (:Transaction {sale_price, sale_date})

// Relationships:
//   (Owner)-[:OWNS]->(Property)
//   (Owner)-[:SOLD {date, price}]->(Property)
//   (Owner)-[:BOUGHT {date, price}]->(Property)
//   (Property)-[:LOCATED_IN]->(Neighborhood)
//   (Owner)-[:CO_INVESTS_WITH]->(Owner)
//   (Property)-[:NEARBY {distance_km}]->(Property)
"""

NEO4J_CYPHER_QUERIES = {
    "create_ownership_graph": """
        // Load ownership relationships from property data
        LOAD CSV WITH HEADERS FROM 'file:///pluto_cleaned.csv' AS row
        MERGE (o:Owner {name: row.ownername})
        MERGE (p:Property {bbl: row.bbl})
        SET p.address = row.address,
            p.borough = row.borough_name,
            p.assessed_value = toFloat(row.assesstot),
            p.latitude = toFloat(row.latitude),
            p.longitude = toFloat(row.longitude)
        MERGE (o)-[:OWNS]->(p)
    """,

    "find_largest_portfolios": """
        // Find owners with the largest portfolios
        MATCH (o:Owner)-[:OWNS]->(p:Property)
        WITH o, COUNT(p) AS n_properties,
             SUM(p.assessed_value) AS portfolio_value
        ORDER BY portfolio_value DESC
        LIMIT 20
        RETURN o.name AS owner, n_properties, portfolio_value
    """,

    "find_co_investment_network": """
        // Find owners who invest in the same neighborhoods
        MATCH (o1:Owner)-[:OWNS]->(p1:Property)-[:LOCATED_IN]->(n:Neighborhood)
              <-[:LOCATED_IN]-(p2:Property)<-[:OWNS]-(o2:Owner)
        WHERE o1 <> o2
        WITH o1, o2, COUNT(DISTINCT n) AS shared_neighborhoods,
             COLLECT(DISTINCT n.name) AS neighborhoods
        WHERE shared_neighborhoods >= 3
        RETURN o1.name, o2.name, shared_neighborhoods, neighborhoods
        ORDER BY shared_neighborhoods DESC
        LIMIT 50
    """,

    "propensity_to_sell_network": """
        // Find properties whose owners have recently sold nearby assets
        MATCH (o:Owner)-[:SOLD]->(p1:Property)
        WHERE p1.sale_date > date() - duration('P2Y')
        WITH o, COUNT(p1) AS recent_sales
        MATCH (o)-[:OWNS]->(p2:Property)
        WHERE NOT (o)-[:SOLD]->(p2)
        RETURN p2.bbl, p2.address, o.name AS owner,
               recent_sales AS owner_recent_sales,
               p2.assessed_value
        ORDER BY recent_sales DESC
    """,

    "neighborhood_investment_flow": """
        // Track capital flow between neighborhoods
        MATCH (buyer:Owner)-[:BOUGHT {date: d}]->(p:Property)
              -[:LOCATED_IN]->(dest:Neighborhood)
        MATCH (seller:Owner)-[:SOLD {date: d}]->(p)
        OPTIONAL MATCH (buyer)-[:OWNS]->(source_prop:Property)
                       -[:LOCATED_IN]->(source:Neighborhood)
        WHERE source <> dest
        RETURN source.name AS from_neighborhood,
               dest.name AS to_neighborhood,
               COUNT(*) AS flow_count,
               SUM(p.assessed_value) AS flow_value
        ORDER BY flow_value DESC
    """,
}


def get_neo4j_connection():
    """Create Neo4j driver connection (optional, for production use)."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {NEO4J_URI}")
        return driver
    except Exception as e:
        logger.warning(f"Neo4j connection not available: {e}")
        return None

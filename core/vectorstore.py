from typing import Any, List, Optional, Tuple, Dict, Union
import uuid
from qdrant_client import QdrantClient, models
from langchain.schema import Document
from core.config import settings
from core.decorators import log_execution


class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
        )

    @log_execution
    def initialize_collections(self):
        """Creates collections with Dense, Sparse, and ColBERT (MaxSim) config."""

        # Configuration for ColBERT (MaxSim)
        # We disable HNSW for ColBERT to save RAM, using it only for Rescoring (Reranking)
        colbert_config = models.VectorParams(
            size=128,  # ColBERT dimension
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=models.HnswConfigDiff(
                m=0
            ),  # Disable Indexing for pure reranking
        )

        dense_config = models.VectorParams(size=384, distance=models.Distance.COSINE)

        for collection in [settings.COLLECTION_ORG, settings.COLLECTION_USER]:
            if not self.client.collection_exists(collection):
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config={
                        "dense": dense_config,
                        "colbert": colbert_config,
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            modifier=models.Modifier.IDF
                        )
                    },
                )

    @log_execution
    def upsert_points(
        self, collection_name: str, points_data: List[Tuple[Document, Dict[str, any]]]
    ):
        """
        Ingests data into Qdrant using named vectors.

        Args:
            collection_name: Target collection.
            points_data: List of (Document, {"dense": d, "sparse": s, "colbert": c})
        """
        points = []

        for doc, embeddings in points_data:
            # 1. Generate a unique ID (UUID) for each chunk
            point_id = str(uuid.uuid4())

            # 2. Extract embeddings from the dictionary
            # Note: Sparse vectors must be formatted as models.SparseVector if not already
            sparse_vec = embeddings.get("sparse")
            if (
                not isinstance(sparse_vec, models.SparseVector)
                and sparse_vec is not None
            ):
                # If 's' is a dict with indices/values or a FastEmbed object
                sparse_vec = models.SparseVector(
                    indices=sparse_vec.get("indices"),
                    values=sparse_vec.get("values"),
                )

            # 3. Construct the PointStruct
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": embeddings.get("dense"),
                        "sparse": sparse_vec,
                        "colbert": embeddings.get("colbert"),
                    },
                    payload={
                        "page_content": doc.page_content,
                        **doc.metadata,  # Flattens LangChain metadata into payload
                    },
                )
            )

        # 4. Perform the upsert in batches for efficiency
        return self.client.upsert(
            collection_name=collection_name, points=points, wait=True
        )

    def create_match_filter(self, key: str, value: str) -> models.Filter:
        """Creates a standard Qdrant exact match filter."""
        return models.Filter(
            must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))]
        )

    def search_single_collection(
        self,
        collection_name: str,
        query_embeddings: Dict[str, Any],
        search_filter: Optional[models.Filter],
        limit: int,
    ) -> List[models.ScoredPoint]:
        """
        Executes the full pipeline on ONE collection:
        Dense + Sparse -> RRF Fusion -> ColBERT Reranking
        """
        # 1. Define Retrieval Limits
        # Fetch 3x candidates to ensure RRF has enough overlap
        retrieval_limit = limit * 3

        # 2. Build Prefetches (Dense & Sparse)
        pf_dense = models.Prefetch(
            query=query_embeddings["dense"],
            using="dense",
            filter=search_filter,
            limit=retrieval_limit,
        )

        pf_sparse = models.Prefetch(
            query=query_embeddings["sparse"],
            using="sparse",
            filter=search_filter,
            limit=retrieval_limit,
        )

        # 3. Build Fusion Prefetch
        # We pass the top 2x candidates to the ColBERT reranker
        fusion_limit = limit * 2
        pf_fusion = models.Prefetch(
            prefetch=[pf_dense, pf_sparse],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fusion_limit,
        )

        # 4. Execute Root Query (ColBERT Rerank)
        # This returns the top 'limit' documents, already reranked.
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                prefetch=pf_fusion,
                query=query_embeddings["colbert"],
                using="colbert",
                limit=limit,
                with_payload=True,
            ).points
            return results
        except Exception as e:
            # Log error but return empty list to not break the loop
            print(f"Error searching {collection_name}: {e}")
            return []

    def search_multiple_collections(
        self,
        query_embeddings: Dict[str, Any],
        search_targets: List[Dict[str, Any]],
        limit: int = 10,
    ) -> List[models.ScoredPoint]:
        """
        Runs search_single_collection for each target, aggregates, and sorts.

        Args:
            search_targets: List of dicts [{"collection": "name", "filter": FilterObj}, ...]
        """
        aggregated_results = []

        # 1. Run Independent Searches
        for target in search_targets:
            collection = target["collection"]
            coll_filter = target.get("filter")

            # We request the full 'limit' from EACH collection to ensure quality
            results = self.search_single_collection(
                collection, query_embeddings, coll_filter, limit
            )
            aggregated_results.extend(results)

        # 2. Global Rerank (Sort)
        # Since ColBERT scores are absolute (MaxSim), we can compare them directly.
        aggregated_results.sort(key=lambda x: x.score, reverse=True)

        # 3. Slice top K
        return aggregated_results[:limit]


### Version 1 below: Previous implementation of hybrid search with single collection only.

# def _build_search_pipeline(
#     self,
#     query_embeddings: Dict[str, any],
#     doc_name_filter: Optional[str],
#     limit: int,
# ) -> Tuple[List[models.Prefetch], any]:
#     """
#     Constructs the Universal Query pipeline:
#     Dense + Sparse -> RRF Fusion -> ColBERT Rerank.
#     """

#     # 1. Construct Filter
#     search_filter = None
#     if doc_name_filter:
#         search_filter = models.Filter(
#             must=[
#                 models.FieldCondition(
#                     key="document_name",
#                     match=models.MatchValue(value=doc_name_filter),
#                 )
#             ]
#         )

#     # 2. Prepare Query Vectors
#     dense_vec = query_embeddings.get("dense")
#     colbert_vec = query_embeddings.get("colbert")

#     # Handle Sparse format
#     sparse_vec = query_embeddings.get("sparse")
#     if not isinstance(sparse_vec, models.SparseVector) and sparse_vec is not None:
#         # Handle case where sparse comes as raw dict or object
#         if hasattr(sparse_vec, "indices") and hasattr(sparse_vec, "values"):
#             sparse_vec = models.SparseVector(
#                 indices=sparse_vec.indices, values=sparse_vec.values
#             )
#         #  elif isinstance(sparse_vec, dict):
#         #      sparse_vec = models.SparseVector(indices=sparse_vec["indices"], values=sparse_vec["values"])
#         else:
#             raise ValueError("Sparse vector format is unrecognized.")

#     # 3. Define Prefetches (The Retrieval & Fusion Stage)

#     # Prefetch A: Dense Search
#     pf_dense = models.Prefetch(
#         query=dense_vec,
#         using="dense",
#         filter=search_filter,
#         limit=limit * 3,
#     )

#     # Prefetch B: Sparse Search
#     pf_sparse = models.Prefetch(
#         query=sparse_vec,
#         using="sparse",
#         filter=search_filter,
#         limit=limit * 3,
#     )

#     # Prefetch C: Fusion (RRF)
#     # This takes the results of A and B, fuses them, and passes them to the root query
#     pf_fusion = models.Prefetch(
#         prefetch=[pf_dense, pf_sparse],
#         query=models.FusionQuery(fusion=models.Fusion.RRF),
#         limit=limit * 2,
#     )

#     # 4. Return the prefetch chain and the Root Query (ColBERT)
#     # The logic is: Use pf_fusion as the source of points, and score them using colbert_vec
#     return pf_fusion, colbert_vec

# @log_execution
# def hybrid_search(
#     self,
#     query_embeddings: Dict[str, any],
#     collection_names: Union[str, List[str]] = [
#         settings.COLLECTION_ORG,
#         settings.COLLECTION_USER,
#     ],
#     doc_name_filter: Optional[str] = None,
#     limit: int = 10,
#     score_threshold: float = 0.0,
# ) -> List[models.ScoredPoint]:
#     """
#     Performs Hybrid Search + RRF + ColBERT Reranking.
#     Can search a single collection or aggregate results from both.

#     Args:
#         query_embeddings: Dict with "dense", "sparse", and "colbert" query vectors.
#         collection_names: String (single) or List of strings (multiple collections).
#         doc_name_filter: Exact match filter for 'document_name' in payload.
#         limit: Number of final results to return.
#         score_threshold: Minimum score (after reranking) to return.

#         Query Structure:
#                     {
#                         "collection_name": "your_collection_name",
#                         "prefetch": [
#                             {
#                             "prefetch": [
#                                 {
#                                 "query": [0.12, 0.45, ...], // Dense Vector
#                                 "using": "dense",
#                                 "filter": { "must": [{ "key": "document_name", "match": { "value": "report_v1.pdf" } }] },
#                                 "limit": 30
#                                 },
#                                 {
#                                 "query": { "indices": [10, 42], "values": [0.8, 0.2] }, // Sparse Vector
#                                 "using": "sparse",
#                                 "filter": { "must": [{ "key": "document_name", "match": { "value": "report_v1.pdf" } }] },
#                                 "limit": 30
#                                 }
#                             ],
#                             "query": { "fusion": "rrf" },
#                             "limit": 20
#                             }
#                         ],
#                         "query": [[0.1, ...], [0.2, ...]], // ColBERT Multi-vector
#                         "using": "colbert",
#                         "limit": 10
#                 }
#     """

#     # Normalize collection_names to a list
#     if isinstance(collection_names, str):
#         target_collections = [collection_names]
#     else:
#         target_collections = collection_names

#     all_results = []

#     # Build the query components
#     pf_fusion, root_query_vec = self._build_search_pipeline(
#         query_embeddings, doc_name_filter, limit
#     )

#     # Execute query against requested collections
#     for collection in target_collections:
#         if not self.client.collection_exists(collection):
#             continue

#         try:
#             results = self.client.query_points(
#                 collection_name=collection,
#                 prefetch=pf_fusion,  # Provide RRF candidates
#                 query=root_query_vec,  # Rescore candidates using ColBERT (MaxSim)
#                 using="colbert",  # Target the colbert vector for the final score
#                 limit=limit,
#                 with_payload=True,
#                 score_threshold=score_threshold,
#             ).points

#             all_results.extend(results)
#         except Exception as e:
#             print(f"Error searching collection {collection}: {e}")

#     # If searching multiple collections, we must re-sort and slice the aggregated results
#     if len(target_collections) > 1:
#         # Sort by score (descending)
#         all_results.sort(key=lambda x: x.score, reverse=True)
#         # Slice to limit
#         all_results = all_results[:limit]

#     return all_results

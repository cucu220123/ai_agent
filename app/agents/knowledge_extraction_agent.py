from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.llm.contracts import extract_json_object


EntityType = Literal["Capability", "Task", "Algorithm", "Dataset", "Feature", "PreprocessingStrategy", "Metric", "Constraint", "Dependency", "Environment", "HyperparameterConfig", "ValidationRun", "FailureExperience", "RepairExperience", "AlgorithmVersion"]


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    type: EntityType
    name: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence_span: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    target: str = Field(min_length=1)
    evidence_span: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class KnowledgeExtractionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    summary: str

    @field_validator("relations")
    @classmethod
    def relation_endpoints_exist(cls, relations: list[ExtractedRelation], info):
        entities = info.data.get("entities", [])
        entity_ids = {entity.id for entity in entities}
        invalid = [rel for rel in relations if rel.source not in entity_ids or rel.target not in entity_ids]
        if invalid:
            raise ValueError("relation endpoints must reference entity ids in the same chunk")
        return relations


class FocusedSemanticContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    new_entities: list[ExtractedEntity] = Field(min_length=1)
    relations: list[ExtractedRelation] = Field(min_length=1)
    summary: str


class KnowledgeExtractionAgent:
    """Chunk-level LLM extraction with strict entities/relations and provenance merge."""

    def __init__(self, llm=None, provider_name: str = "mock", max_attempts: int = 3):
        self.llm = llm
        self.provider_name = provider_name
        self.max_attempts = max_attempts

    def run(self, source_path: str | Path, deterministic_facts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        source_path = str(source_path)
        chunks = deterministic_facts.get("content_chunks") or [{"chunk_id": "facts", "heading": "deterministic facts", "text": json.dumps(deterministic_facts, ensure_ascii=False)}]
        trace: dict[str, Any] = {"provider": self.provider_name, "status": "deterministic_fallback", "source": source_path, "chunk_count": len(chunks), "chunks": []}
        accepted: list[KnowledgeExtractionContract] = []
        if self.llm is not None and self.provider_name != "mock":
            for chunk in chunks:
                previous, last_error, attempts, partial_contract = "", "", [], None
                for attempt in range(1, self.max_attempts + 1):
                    try:
                        payload = {
                            "source": source_path, "chunk": chunk,
                            "ast_facts": {k: v for k, v in deterministic_facts.items() if k not in {"content_chunks", "functions", "classes"}},
                            "allowed_entity_types_exact": ["Capability", "Task", "Algorithm", "Dataset", "Feature", "PreprocessingStrategy", "Metric", "Constraint", "Dependency", "Environment", "HyperparameterConfig", "ValidationRun", "FailureExperience", "RepairExperience", "AlgorithmVersion"],
                            "required_schema": {"entities": [{"id": "feature_age", "type": "Feature", "name": "age", "properties": {"dtype": "numeric"}, "evidence_span": "age | 数值 | 客户年龄", "confidence": 0.9}], "relations": [{"source": "capability_id", "relation": "USES_ALGORITHM", "target": "algorithm_id", "evidence_span": "exact source fragment", "confidence": 0.9}], "summary": "grounded chunk summary"},
                            "top_level_keys_exact": ["entities", "relations", "summary"],
                            "rules": ["top-level JSON must contain exactly entities, relations, summary; do not add metrics/features keys", "put every metric and feature inside entities", "type must use an exact English value from allowed_entity_types_exact", "confidence is required for every entity and relation", "relation endpoints must reference ids from entities", "evidence_span must be copied from the chunk", "return empty relations if no relation is stated"],
                            "mandatory_observations": {"known_metrics": deterministic_facts.get("metrics", []), "known_fields": deterministic_facts.get("fields", []), "headings": deterministic_facts.get("headings", []), "document_kind": deterministic_facts.get("kind")},
                        }
                        if attempt > 1:
                            payload.update({"repair_previous_output": previous[:10000], "validation_error": last_error})
                        if attempt == self.max_attempts and deterministic_facts.get("kind") in {"markdown", "validation_report"}:
                            payload["focused_semantic_completion"] = {
                                "must_include_entity_types": ["Capability", "Task", "Metric"],
                                "must_include_metric_names": deterministic_facts.get("metrics", []),
                                "derive_capability_and_task_from_headings": deterministic_facts.get("headings", []),
                                "must_include_relations_when_grounded": ["SOLVES", "EVALUATED_BY"],
                                "instruction": "Rebuild the entire JSON. Include the prior valid Feature/Metric entities, then add source-grounded Capability/Task and relations. Do not add unsupported algorithms.",
                            }
                        raw = self.llm.complete("你是 KnowledgeExtractionAgent。只返回一个严格 JSON 对象，不得杜撰，不要 Markdown/思考过程。", json.dumps(payload, ensure_ascii=False), purpose="extraction") or ""
                        previous = raw
                        contract = KnowledgeExtractionContract.model_validate(extract_json_object(raw))
                        contract, semantic_corrections = self._ground_contract(contract, deterministic_facts)
                        partial_contract = contract
                        quality_errors = self._quality_gate(contract, deterministic_facts, chunk)
                        if quality_errors:
                            raise ValueError("; ".join(quality_errors))
                        accepted.append(contract)
                        attempts.append({"attempt": attempt, "status": "accepted"})
                        trace["chunks"].append({"chunk_id": chunk.get("chunk_id"), "status": "accepted", "attempts": attempts, "semantic_corrections": semantic_corrections, "entities": len(contract.entities), "relations": len(contract.relations), "token_usage": getattr(self.llm, "last_usage", {}), "generation": getattr(self.llm, "last_generation", {})})
                        break
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"[:1800]
                        attempts.append({"attempt": attempt, "status": "rejected", "error": last_error})
                else:
                    focused = self._focused_completion(source_path, chunk, deterministic_facts, partial_contract)
                    if focused is not None:
                        accepted.append(focused)
                        attempts.append({"attempt": "focused_semantic_completion", "status": "accepted"})
                        trace["chunks"].append({"chunk_id": chunk.get("chunk_id"), "status": "accepted", "attempts": attempts, "entities": len(focused.entities), "relations": len(focused.relations), "token_usage": getattr(self.llm, "last_usage", {}), "generation": getattr(self.llm, "last_generation", {})})
                    else:
                        trace["chunks"].append({"chunk_id": chunk.get("chunk_id"), "status": "rejected", "attempts": attempts, "error": last_error, "generation": getattr(self.llm, "last_generation", {})})
        if accepted:
            merged = self._merge(accepted, source_path)
            trace.update({"status": "ok", "accepted_chunks": len(accepted), "rejected_chunks": len(chunks) - len(accepted)})
            return merged, trace
        fallback = self._deterministic_fallback(deterministic_facts, source_path)
        trace["fallback_reason"] = "no chunk passed strict extraction contract"
        return fallback, trace

    def _focused_completion(self, source_path: str, chunk: dict[str, Any], facts: dict[str, Any], partial: KnowledgeExtractionContract | None) -> KnowledgeExtractionContract | None:
        if partial is None or facts.get("kind") not in {"markdown", "validation_report"}:
            return None
        known = [entity.model_dump() for entity in partial.entities]
        try:
            raw = self.llm.complete(
                "只输出 JSON：new_entities, relations, summary。不要重复 known_entities。",
                json.dumps({
                    "source_text": chunk.get("text", ""), "known_entities": known,
                    "required": {"new_entities": ["one Capability and one Task with exact English type, evidence_span, confidence"], "relations": ["Capability SOLVES Task", "Capability EVALUATED_BY each known Metric when stated"], "summary": "string"},
                    "rule": "relation source/target must use ids from known_entities or new_entities; evidence_span copied from source",
                }, ensure_ascii=False),
                purpose="extraction",
            ) or ""
            focused = FocusedSemanticContract.model_validate(extract_json_object(raw))
            new_types = {entity.type for entity in focused.new_entities}
            if not {"Capability", "Task"}.issubset(new_types):
                return None
            all_entities = [*partial.entities, *focused.new_entities]
            ids = {entity.id for entity in all_entities}
            if any(relation.source not in ids or relation.target not in ids for relation in focused.relations):
                return None
            merged = KnowledgeExtractionContract(entities=all_entities, relations=[*partial.relations, *focused.relations], summary=focused.summary)
            return merged if not self._quality_gate(merged, facts, chunk) else None
        except Exception:
            return None

    @staticmethod
    def _quality_gate(contract: KnowledgeExtractionContract, facts: dict[str, Any], chunk: dict[str, Any]) -> list[str]:
        errors = []
        entity_types = {entity.type for entity in contract.entities}
        metric_names = {entity.name.lower().replace("-", "_") for entity in contract.entities if entity.type == "Metric"}
        expected_metrics = {str(metric).lower().replace("-", "_") for metric in facts.get("metrics", [])}
        if expected_metrics and not expected_metrics.issubset(metric_names):
            errors.append(f"missing known Metric entities: {sorted(expected_metrics - metric_names)}")
        if facts.get("kind") in {"markdown", "validation_report"} and facts.get("headings") and not ({"Capability", "Task"} & entity_types):
            errors.append("document-level extraction requires Capability or Task")
        text = chunk.get("text", "")
        if any(token in text.lower() for token in ("预测", "使用", "requires", "uses", "evaluated", "指标")) and not contract.relations:
            errors.append("explicit relationships in source require at least one grounded relation")
        return errors

    @staticmethod
    def _ground_contract(contract: KnowledgeExtractionContract, facts: dict[str, Any]) -> tuple[KnowledgeExtractionContract, list[str]]:
        fields = {str(value).lower() for value in facts.get("fields", [])}
        metrics = {str(value).lower().replace("-", "_") for value in facts.get("metrics", [])}
        corrections = []
        payload = contract.model_dump()
        for entity in payload["entities"]:
            normalized = str(entity["name"]).lower().replace("-", "_")
            if normalized in fields and entity["type"] != "Feature":
                corrections.append(f"{entity['id']}: type {entity['type']} -> Feature from deterministic field evidence")
                entity["type"] = "Feature"
            elif normalized in metrics and entity["type"] != "Metric":
                corrections.append(f"{entity['id']}: type {entity['type']} -> Metric from deterministic metric evidence")
                entity["type"] = "Metric"
        return KnowledgeExtractionContract.model_validate(payload), corrections

    @staticmethod
    def _merge(contracts: list[KnowledgeExtractionContract], source_path: str) -> dict[str, Any]:
        entities: dict[tuple[str, str], dict[str, Any]] = {}
        relations: dict[tuple[str, str, str], dict[str, Any]] = {}
        summaries = []
        for contract in contracts:
            summaries.append(contract.summary)
            for entity in contract.entities:
                payload = entity.model_dump()
                payload["provenance"] = {"source": source_path, "evidence_span": entity.evidence_span}
                key = (entity.type, entity.name.lower())
                if key not in entities or entity.confidence > entities[key]["confidence"]:
                    entities[key] = payload
            for relation in contract.relations:
                payload = relation.model_dump()
                payload["provenance"] = {"source": source_path, "evidence_span": relation.evidence_span}
                key = (relation.source, relation.relation, relation.target)
                if key not in relations or relation.confidence > relations[key]["confidence"]:
                    relations[key] = payload
        return {"entities": list(entities.values()), "relations": list(relations.values()), "summary": "\n".join(dict.fromkeys(summaries)), "provenance": {"source": source_path}}

    @staticmethod
    def _deterministic_fallback(facts: dict[str, Any], source_path: str) -> dict[str, Any]:
        entities = []
        for metric in facts.get("metrics", []):
            entities.append({"id": f"metric_{str(metric).lower().replace('-', '_')}", "type": "Metric", "name": str(metric), "properties": {}, "evidence_span": str(metric), "confidence": 0.7, "provenance": {"source": source_path, "evidence_span": str(metric)}})
        for field_name in facts.get("fields", []):
            entities.append({"id": f"feature_{field_name}", "type": "Feature", "name": field_name, "properties": {}, "evidence_span": field_name, "confidence": 0.7, "provenance": {"source": source_path, "evidence_span": field_name}})
        return {"entities": entities, "relations": [], "summary": facts.get("summary", "deterministic extraction"), "provenance": {"source": source_path}, "deterministic_facts": facts}

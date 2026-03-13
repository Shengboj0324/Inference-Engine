"""Inference pipeline orchestrator.

This module orchestrates the complete inference pipeline:
1. Normalization: RawObservation -> NormalizedObservation
2. Candidate Retrieval: Find similar signals
3. LLM Adjudication: Structured signal classification
4. Calibration: Probability calibration
5. Abstention: Decide whether to abstain

This is the main entry point for the Phase 2 inference system.
"""

import logging
from typing import Optional
from uuid import UUID

from app.domain.raw_models import RawObservation
from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalInference
from app.intelligence.normalization import NormalizationEngine
from app.intelligence.candidate_retrieval import CandidateRetriever
from app.intelligence.llm_adjudicator import LLMAdjudicator
from app.intelligence.calibration import Calibrator
from app.intelligence.abstention import AbstentionDecider

logger = logging.getLogger(__name__)


class InferencePipeline:
    """End-to-end inference pipeline orchestrator.
    
    Chains together all Phase 2 components:
    - Normalization
    - Candidate retrieval
    - LLM adjudication
    - Calibration
    - Abstention
    """
    
    def __init__(
        self,
        normalization_engine: Optional[NormalizationEngine] = None,
        candidate_retriever: Optional[CandidateRetriever] = None,
        llm_adjudicator: Optional[LLMAdjudicator] = None,
        calibrator: Optional[Calibrator] = None,
        abstention_decider: Optional[AbstentionDecider] = None,
    ):
        """Initialize inference pipeline.
        
        Args:
            normalization_engine: Normalization engine (creates default if None)
            candidate_retriever: Candidate retriever (creates default if None)
            llm_adjudicator: LLM adjudicator (creates default if None)
            calibrator: Calibrator (creates default if None)
            abstention_decider: Abstention decider (creates default if None)
        """
        self.normalization_engine = normalization_engine or NormalizationEngine(
            enable_translation=False,  # Disable for performance
            enable_entity_extraction=False,  # Disable if spacy not available
            enable_embedding_generation=True,
        )
        
        self.candidate_retriever = candidate_retriever or CandidateRetriever(
            top_k=5
        )
        
        self.llm_adjudicator = llm_adjudicator or LLMAdjudicator(
            model_name="gpt-4-turbo",
            temperature=0.3,
        )
        
        self.calibrator = calibrator or Calibrator(
            method="temperature"
        )
        
        self.abstention_decider = abstention_decider or AbstentionDecider()
        
        logger.info("InferencePipeline initialized with all components")
    
    async def run(
        self,
        raw_observation: RawObservation,
        skip_normalization: bool = False,
        normalized_observation: Optional[NormalizedObservation] = None,
    ) -> tuple[NormalizedObservation, SignalInference]:
        """Run the complete inference pipeline.
        
        Args:
            raw_observation: Raw observation from connector
            skip_normalization: Skip normalization if already done
            normalized_observation: Pre-normalized observation (if skip_normalization=True)
            
        Returns:
            Tuple of (normalized_observation, signal_inference)
        """
        logger.info(f"Running inference pipeline for observation {raw_observation.id}")
        
        # Stage 1: Normalization
        if skip_normalization and normalized_observation:
            normalized = normalized_observation
            logger.debug("Skipped normalization (using provided)")
        else:
            logger.debug("Stage 1: Normalization")
            normalized = await self.normalization_engine.normalize(raw_observation)
        
        # Stage 2: Candidate Retrieval
        logger.debug("Stage 2: Candidate Retrieval")
        candidates = await self.candidate_retriever.retrieve_candidates(normalized)
        logger.debug(f"Retrieved {len(candidates)} candidates")
        
        # Stage 3: LLM Adjudication
        logger.debug("Stage 3: LLM Adjudication")
        inference = await self.llm_adjudicator.adjudicate(normalized, candidates)
        logger.debug(f"LLM adjudication complete: abstained={inference.abstained}")
        
        # Stage 4: Calibration
        logger.debug("Stage 4: Calibration")
        inference = self.calibrator.calibrate(inference)
        logger.debug("Calibration complete")
        
        # Stage 5: Abstention Decision
        logger.debug("Stage 5: Abstention Decision")
        should_abstain, reason, explanation = self.abstention_decider.should_abstain(
            inference, normalized, candidates
        )
        
        if should_abstain and not inference.abstained:
            # Override inference with abstention
            inference.abstained = True
            inference.abstention_reason = reason
            inference.rationale = f"{inference.rationale or ''}\n\nAbstention: {explanation}"
            logger.debug(f"Abstention triggered: {reason.value if reason else 'unknown'}")
        
        logger.info(
            f"Pipeline complete for {raw_observation.id}: "
            f"signal={inference.top_prediction.signal_type.value if inference.top_prediction else 'none'}, "
            f"confidence={inference.top_prediction.probability if inference.top_prediction else 0.0:.2f}, "
            f"abstained={inference.abstained}"
        )
        
        return normalized, inference
    
    async def run_batch(
        self,
        raw_observations: list[RawObservation],
    ) -> list[tuple[NormalizedObservation, SignalInference]]:
        """Run pipeline on a batch of observations.
        
        Args:
            raw_observations: List of raw observations
            
        Returns:
            List of (normalized_observation, signal_inference) tuples
        """
        results = []
        for raw_obs in raw_observations:
            try:
                result = await self.run(raw_obs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing observation {raw_obs.id}: {e}", exc_info=True)
                # Continue with next observation
        
        logger.info(f"Batch processing complete: {len(results)}/{len(raw_observations)} successful")
        return results


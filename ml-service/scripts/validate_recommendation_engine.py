"""
Celest Machine Learning Subsystem - System Integration & Validation Suite.
Executes comprehensive end-to-end black-box integration validation against the FastAPI REST server.
"""

import gc
import json
import logging
import math
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Callable

# Global Configuration Constants
API_BASE_URL: str = "http://127.0.0.1:8000"
HTTP_TIMEOUT_SECONDS: float = 10.0
MAX_STRESS_TEST_SAMPLE_SIZE: int = 50
DEFAULT_TOP_K: int = 10
LATENCY_SAMPLES_COUNT: int = 5
CACHE_WARM_SAMPLES_COUNT: int = 4
MAX_ALLOWED_DIVERSITY_OVERLAP_FRACTION: float = 0.90

# Granular Latency Threshold Constraints (in milliseconds)
MAX_HEALTH_LATENCY_MS: float = 50.0
MAX_STATS_LATENCY_MS: float = 50.0
MAX_RECOMMENDATION_LATENCY_MS: float = 250.0

# Fallback Track Configuration
DEFAULT_TRACK_ID_FALLBACK: List[int] = [2, 3, 5, 10, 20, 30, 140, 141, 142, 182, 1095]

# Configure structured, production-ready logging outputs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("celest_validation_suite")


def log_pass(message: str) -> None:
    """Standardized logger wrapper for reporting successful validation checks."""
    logger.info(f"[PASS] {message}")


def log_fail(message: str) -> None:
    """Standardized logger wrapper for reporting failed validation boundaries."""
    logger.error(f"[FAIL] {message}")


def log_info(message: str) -> None:
    """Standardized logger wrapper for reporting informational progress data."""
    logger.info(f"[INFO] {message}")


# Gracefully import colorama for production terminal status formatting
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLOR_PASS = f"{Fore.GREEN}{Style.BRIGHT}PASS{Style.RESET_ALL}"
    COLOR_FAIL = f"{Fore.RED}{Style.BRIGHT}FAIL{Style.RESET_ALL}"
except ImportError:
    COLOR_PASS = "PASS"
    COLOR_FAIL = "FAIL"

# Import external dependency validation layer
try:
    import requests
    import pydantic
    import fastapi
except ImportError as e:
    logger.critical(f"Missing core system dependency: {e}. Package environment verification failed.")
    sys.exit(1)


class CelestEngineValidator:
    """
    Enterprise-grade integration QA controller that systematically executes black-box 
    assertions, statistical diversity checks, and performance benchmarking against the active Celest API.
    """

    def __init__(self, base_url: str = API_BASE_URL) -> None:
        """Initializes the verification coordinator bound to the API root address."""
        self.base_url = base_url.rstrip("/")
        
        # Latency matrix tracking maps storing raw float seconds captures
        self.latency_metrics: Dict[str, List[float]] = {
            "health": [],
            "stats": [],
            "recommendation": []
        }
        
        self.cached_valid_track_ids: List[int] = []
        
        # Test Case Dispatch Matrix Mapping Name to Test Function
        self.test_cases: Dict[str, Callable[[], bool]] = {
            "API Reachability": self.validate_api_reachability,
            "Health Endpoint": self.validate_health_endpoint,
            "Statistics Endpoint": self.validate_stats_endpoint,
            "Recommendation Endpoint": self._run_recommendation_pipeline_checks,
            "Recommendation Cache": self.validate_cache_efficiency,
            "Recommendation Schema": lambda: self._last_rec_schema_passed,
            "Score Validation": lambda: self._last_rec_scores_passed,
            "Ordering": lambda: self._last_rec_ordering_passed,
            "Duplicate Detection": lambda: self._last_rec_dedup_passed,
            "404 Handling": self.validate_error_handling_404,
            "422 Handling": self.validate_error_handling_422,
            "Latency Benchmark": self.validate_latency_benchmarks,
            "Stress Test": self.validate_random_stress_and_diversity
        }
        
        # Dynamic internal trackers updated during processing blocks
        self._last_rec_schema_passed: bool = False
        self._last_rec_scores_passed: bool = False
        self._last_rec_ordering_passed: bool = False
        self._last_rec_dedup_passed: bool = False
        
        # Richer reporting statistics
        self._cached_cold_display: float = 0.0
        self._cached_avg_warm_display: float = 0.0
        self._cached_avg_overlap_pct: float = 0.0
        self._cached_max_overlap_pct: float = 0.0
        
        self.test_results: Dict[str, Optional[bool]] = {name: None for name in self.test_cases}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[requests.Response, float]:
        """Executes an HTTP GET request and returns the response alongside its network latency context."""
        url = f"{self.base_url}{path}"
        start_time = time.perf_counter()
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        duration = time.perf_counter() - start_time
        return response, duration

    def _get_fallback_track_inventory(self) -> List[int]:
        """
        Isolated helper function supplying verified historical track identifiers.
        Acts as an engineering fallback if specific indexing catalog endpoints are decoupled.
        """
        logger.debug("Retrieving deterministic track inventory from internal fallback matrix.")
        return DEFAULT_TRACK_ID_FALLBACK.copy()

    def _hydrate_valid_tracks_inventory(self) -> None:
        """
        Hydrates the active testing pool registry from the live server endpoint.
        Gracefully handles both raw list outputs and dictionary structured payloads.
        """
        try:
            # Attempt to pull track array list from future production inventory endpoint
            res, _ = self._get("/recommendations/tracks")
            if res.status_code == 200:
                payload = res.json()
                parsed_ids: Optional[List[int]] = None

                # Format A: Direct payload array response [2, 5, 10]
                if isinstance(payload, list):
                    parsed_ids = [int(t) for t in payload]
                # Format B: Object enveloped track array response {"track_ids": [2, 5, 10]}
                elif isinstance(payload, dict) and "track_ids" in payload and isinstance(payload["track_ids"], list):
                    parsed_ids = [int(t) for t in payload["track_ids"]]

                if parsed_ids:
                    self.cached_valid_track_ids = parsed_ids
                    log_info(f"Dynamically discovered {len(self.cached_valid_track_ids)} tracks from live inventory route.")
                    return
        except Exception as err:
            logger.debug(f"Direct track inventory route unreachable or unprovisioned: {err}")
        
        # NOTE: The DEFAULT_TRACK_ID_FALLBACK array is an internal placeholder.
        # It operates exclusively as a fallback mechanism for localized development contexts
        # because the production server architecture has not yet implemented a '/recommendations/tracks' 
        # inventory tracking endpoint. As soon as that endpoint becomes online, this fallback layer 
        # will automatically handle both list and wrapped dict definitions without requiring code changes.
        logger.warning("Track inventory path '/recommendations/tracks' is unprovisioned. Ingesting static default fallback keys.")
        self.cached_valid_track_ids = self._get_fallback_track_inventory()

    def validate_api_reachability(self) -> bool:
        """Verifies that the target FastAPI deployment context is alive and accepting connections."""
        log_info("Executing API Root Reachability Check...")
        try:
            res, _ = self._get("/")
            if res.status_code == 200:
                log_pass("API Root responded successfully with HTTP 200.")
                return True
            log_fail(f"API Root returned unexpected status code: {res.status_code}")
            return False
        except Exception as err:
            logger.error(f"Reachability assertion encountered system crash context: {err}")
            log_fail("API Server is completely unreachable.")
            return False

    def validate_health_endpoint(self) -> bool:
        """Audits the technical layout variables presented inside the system availability schemas."""
        log_info("Executing Health Subsystem Report Verification...")
        try:
            res, duration = self._get("/recommendations/health")
            self.latency_metrics["health"].append(duration)
            if res.status_code != 200:
                log_fail(f"Health endpoint failed with unexpected status code: {res.status_code}")
                return False
            
            payload = res.json()
            checks = (
                payload.get("status") == "healthy" and
                payload.get("engine_loaded") is True and
                payload.get("vectors", 0) > 0 and
                payload.get("tracks", 0) > 0 and
                payload.get("cached_audio_features", 0) > 0
            )
            if checks:
                log_pass("Health response keys conform to valid technical runtime parameters.")
                return True
            log_fail(f"Health payload parameters are broken or unhydrated: {payload}")
            return False
        except Exception as err:
            logger.error(f"Health diagnostics validation step failed with crash sequence: {err}")
            log_fail("Health endpoint validation execution error.")
            return False

    def validate_stats_endpoint(self) -> bool:
        """Validates consistency between index vectors, relational track maps, and feature cache bounds."""
        log_info("Executing Internal Operational Statistics Audit...")
        try:
            res, duration = self._get("/recommendations/stats")
            self.latency_metrics["stats"].append(duration)
            if res.status_code != 200:
                log_fail(f"Stats endpoint failed with unexpected status code: {res.status_code}")
                return False
            
            payload = res.json()
            checks = (
                payload.get("vectors") == payload.get("track_mapping") and
                payload.get("embedding_dimension", 0) > 0 and
                payload.get("candidate_pool_size", 0) > 0 and
                payload.get("audio_cache", 0) > 0
            )
            if checks:
                log_pass("System statistics variables provide accurate matrix structure alignment.")
                return True
            log_fail(f"System metrics mismatch or negative value limits detected: {payload}")
            return False
        except Exception as err:
            logger.error(f"Statistics ledger check failed with structural error: {err}")
            log_fail("Statistics validation endpoint processing error.")
            return False

    def _validate_individual_track_item(self, item: Dict[str, Any]) -> bool:
        """Enforces schema type and content quality conformity checks across individual prediction elements."""
        try:
            assert "track_id" in item and isinstance(item["track_id"], int), "Malformed track_id type representation."
            
            # Metadata Quality Assertions: Ensure strings exist, are populated, and contain real context
            assert "title" in item and str(item["title"]).strip() != "", "Song title string key is missing or entirely blank."
            assert "artist" in item and str(item["artist"]).strip() != "", "Artist parameter is missing or entirely blank."
            assert "album" in item and str(item["album"]).strip() != "", "Album descriptor element key is missing or entirely blank."
            
            assert item["title"] != "Unknown", "Acoustic tracking failure: Metadata title resolved to fallback placeholder 'Unknown'."
            assert item["artist"] != "Unknown", "Acoustic tracking failure: Metadata artist resolved to fallback placeholder 'Unknown'."
            
            for key in ["hybrid_score", "semantic_score", "audio_score"]:
                assert key in item and isinstance(item[key], (float, int)), f"Metric '{key}' is not a numeric scalar format."
            return True
        except AssertionError as err:
            logger.debug(f"Track schematic content or data quality evaluation failure: {err}")
            return False

    def _validate_vector_score_boundaries(self, item: Dict[str, Any]) -> bool:
        """Asserts that similarity and hybrid metrics conform strictly to finite standard probability ranges."""
        try:
            for key in ["hybrid_score", "semantic_score", "audio_score"]:
                val = float(item[key])
                
                # Numeric Stability Auditing: Rejects NaN, Positive Infinity, and Negative Infinity
                assert math.isfinite(val), f"Numerical Instability Hazard: Field '{key}' calculated to non-finite representation: {val}"
                assert 0.0 <= val <= 1.0, f"Value probability boundary break for '{key}': {val}"
            return True
        except (AssertionError, ValueError):
            return False

    def _run_recommendation_pipeline_checks(self) -> bool:
        """Orchestrates structural assertions across a dynamically selected inference query route."""
        log_info("Executing Recommendation Pipeline Verification Suite...")
        
        self._hydrate_valid_tracks_inventory()
        if not self.cached_valid_track_ids:
            log_fail("Recommendation pipeline blocked: Valid tracking inventory is empty.")
            return False
            
        track_id = random.choice(self.cached_valid_track_ids)
        log_info(f"Selected random query track anchor ID: {track_id}")

        try:
            res, duration = self._get(f"/recommendations/{track_id}", params={"top_k": DEFAULT_TOP_K})
            self.latency_metrics["recommendation"].append(duration)
            if res.status_code != 200:
                log_fail(f"Recommendation route failed with status code: {res.status_code}")
                return False
            
            payload = res.json()
            assert "query_track" in payload, "Response object is missing core 'query_track' DTO mapping."
            assert payload["query_track"].get("track_id") == track_id, "Query tracking echo ID mismatch."
            
            recs = payload.get("recommendations", [])
            assert len(recs) == DEFAULT_TOP_K, f"Output sizing constraint error. Expected {DEFAULT_TOP_K}, received {len(recs)}"

            self._last_rec_schema_passed = True
            self._last_rec_scores_passed = True
            self._last_rec_ordering_passed = True
            self._last_rec_dedup_passed = True

            previous_hybrid_score = 2.0
            observed_track_ids: Set[int] = set()

            for rec in recs:
                if not self._validate_individual_track_item(rec):
                    self._last_rec_schema_passed = False
                
                if not self._validate_vector_score_boundaries(rec):
                    self._last_rec_scores_passed = False

                current_hybrid_score = float(rec.get("hybrid_score", 0.0))
                if current_hybrid_score > previous_hybrid_score:
                    self._last_rec_ordering_passed = False
                previous_hybrid_score = current_hybrid_score

                rec_track_id = int(rec.get("track_id", -1))
                if rec_track_id in observed_track_ids:
                    self._last_rec_dedup_passed = False
                observed_track_ids.add(rec_track_id)

                assert rec_track_id != track_id, "Self-recommendation guard violation. Query track leaked into results."

            pipeline_checks = all([
                self._last_rec_schema_passed,
                self._last_rec_scores_passed,
                self._last_rec_ordering_passed,
                self._last_rec_dedup_passed
            ])
            
            if pipeline_checks:
                log_pass("Recommendation endpoint processing suite completed without constraints.")
                return True
                
            log_fail("Internal constraints tracking failed across recommendation array fields.")
            return False
        except Exception as err:
            logger.error(f"[!] Primary recommendation verification checks threw exception states: {err}")
            log_fail("Recommendation pipeline validation encountered operational crash.")
            return False

    def validate_cache_efficiency(self) -> bool:
        """
        Verifies the caching performance profile of consecutive queries.
        Ignores the cold load and averages 4 sequential warm hits to verify optimization.
        """
        log_info("Executing Statistical Cache Efficiency Verification Test...")
        if not self.cached_valid_track_ids:
            self._hydrate_valid_tracks_inventory()
            
        if not self.cached_valid_track_ids:
            log_fail("Cache validation skipped: Valid track list empty.")
            return False
            
        track_id = random.choice(self.cached_valid_track_ids)
        try:
            # 1. Execute primary cold request, isolating it from warm cache measurements
            _, duration_cold = self._get(f"/recommendations/{track_id}", params={"top_k": DEFAULT_TOP_K})
            
            # 2. Execute consecutive warm lookup tracking loops
            warm_latencies = []
            for _ in range(CACHE_WARM_SAMPLES_COUNT):
                _, duration_warm = self._get(f"/recommendations/{track_id}", params={"top_k": DEFAULT_TOP_K})
                warm_latencies.append(duration_warm)
                
            avg_warm_latency = sum(warm_latencies) / len(warm_latencies)
            
            # Save measurements to runtime diagnostics cache maps
            self._cached_cold_display = duration_cold * 1000
            self._cached_avg_warm_display = avg_warm_latency * 1000
            
            log_info(f" -> Recorded Isolated Cold Latency: {self._cached_cold_display:.2f} ms")
            log_info(f" -> Computed Average Warm Latency:  {self._cached_avg_warm_display:.2f} ms")
            
            # Assert warm cache averages do not surpass the cold performance bounds threshold
            if avg_warm_latency <= (duration_cold * 1.20):
                log_pass("In-memory tracking loops verified. Warm cache matrix requests execute optimally.")
                return True
                
            log_fail("Local environment caching overhead bounds exceeded. Efficiency constraints broken.")
            return False
        except Exception as err:
            logger.error(f"[!] Cache tracking metrics check encountered an operational exception: {err}")
            log_fail("Recommendation cache efficiency evaluation failure.")
            return False

    def validate_error_handling_404(self) -> bool:
        """Verifies that queries for unindexed track identifiers return an explicit HTTP 404 response payload."""
        log_info("Executing Non-Existent Track Identity 404 Boundary Check...")
        try:
            invalid_track_id = 999999999
            res, _ = self._get(f"/recommendations/{invalid_track_id}")
            if res.status_code == 404:
                log_pass("Server handled unindexed ID request correctly, returning HTTP 404.")
                return True
            log_fail(f"Expected HTTP 404 status intercept, received: {res.status_code}")
            return False
        except Exception as err:
            logger.error(f"[!] Defensive 404 exception handling verification failed: {err}")
            log_fail("404 boundary handling verification failure.")
            return False

    def validate_error_handling_422(self) -> bool:
        """Verifies that query parameters outside the allowed boundaries (1-100) trigger HTTP 422 validation errors."""
        log_info("Executing Parameter Range Boundary Constraint 422 Check...")
        try:
            res_low, _ = self._get("/recommendations/2", params={"top_k": 0})
            res_high, _ = self._get("/recommendations/2", params={"top_k": 101})
            
            if res_low.status_code == 422 and res_high.status_code == 422:
                log_pass("Out-of-bounds top_k queries were rejected with HTTP 422.")
                return True
            log_fail(f"Parameter guards failed. HTTP statuses: top_k=0 ({res_low.status_code}), top_k=101 ({res_high.status_code})")
            return False
        except Exception as err:
            logger.error(f"[!] Parameter validation error checking failed: {err}")
            log_fail("422 validation parameter guard testing failure.")
            return False

    def validate_latency_benchmarks(self) -> bool:
        """Evaluates remaining latency iterations and computes min, max, mean, and standard deviation profile rows."""
        log_info("Executing Statistical Multi-Sample Latency Benchmark Metrics Check...")
        try:
            # Complete sampling count limits up to the required multi-sample footprint threshold
            for _ in range(LATENCY_SAMPLES_COUNT - 1):
                _, d_h = self._get("/recommendations/health")
                self.latency_metrics["health"].append(d_h)
                
                _, d_s = self._get("/recommendations/stats")
                self.latency_metrics["stats"].append(d_s)
                
                t_id = random.choice(self.cached_valid_track_ids) if self.cached_valid_track_ids else 2
                _, d_r = self._get(f"/recommendations/{t_id}", params={"top_k": DEFAULT_TOP_K})
                self.latency_metrics["recommendation"].append(d_r)
                
            global_latency_pass = True
            
            # Map metrics endpoints to trace and compute latency ranges
            endpoint_thresholds = [
                ("health", MAX_HEALTH_LATENCY_MS, "Health Endpoint"),
                ("stats", MAX_STATS_LATENCY_MS, "Statistics Endpoint"),
                ("recommendation", MAX_RECOMMENDATION_LATENCY_MS, "Recommendation Endpoint")
            ]
            
            print("\n" + "-" * 85)
            print("                       ENTERPRISE ENDPOINT LATENCY PROFILES")
            print("-" * 85)
            
            for key, max_allowed, descriptive_label in endpoint_thresholds:
                raw_ms = [val * 1000 for val in self.latency_metrics[key]]
                min_ms = min(raw_ms)
                max_ms = max(raw_ms)
                avg_ms = sum(raw_ms) / len(raw_ms)
                
                # Compute statistical Standard Deviation for variance analytics
                std_ms = statistics.stdev(raw_ms) if len(raw_ms) > 1 else 0.0
                
                print(f" {descriptive_label}:")
                print(f"   ↳ Min: {min_ms:.2f} ms | Max: {max_ms:.2f} ms | Avg: {avg_ms:.2f} ms | StdDev: {std_ms:.2f} ms")
                
                if avg_ms > max_allowed:
                    logger.warning(f"   [!] Performance Alert: Average {key} latency ({avg_ms:.2f} ms) exceeds threshold ({max_allowed} ms)")
                    global_latency_pass = False
                    
            print("-" * 85)
            
            if global_latency_pass:
                log_pass("All statistical average latency bounds run successfully inside performance limits.")
                return True
                
            log_fail("One or more average route latency profiles broke performance threshold lines.")
            return False
        except Exception as err:
            logger.error(f"[!] Latency evaluation check triggered errors: {err}")
            log_fail("Latency threshold evaluation processing error.")
            return False

    def validate_random_stress_and_diversity(self) -> bool:
        """
        Executes a deduplicated randomized stress test and performs mathematically rigorous pairwise matrix 
        overlap analysis via set intersections to identify prediction collapse states.
        """
        log_info("Executing Jaccard Pairwise Intersect Matrix Diversity & Stress Test...")
        if not self.cached_valid_track_ids:
            self._hydrate_valid_tracks_inventory()
        
        if not self.cached_valid_track_ids:
            log_fail("Stress Test Skip: Operational tracking inventory could not be verified.")
            return False

        # Compute adaptive unique testing profile boundaries
        sample_size = min(MAX_STRESS_TEST_SAMPLE_SIZE, len(self.cached_valid_track_ids))
        logger.info(f"Calculated unique stress pool sampling allocation configuration size: {sample_size}")

        # Enforce unique random sample generation to guarantee zero track selection repetitions
        sampled_query_track_ids = random.sample(self.cached_valid_track_ids, sample_size)

        try:
            all_observed_prediction_sets: List[Set[int]] = []

            for iteration, random_track_id in enumerate(sampled_query_track_ids, start=1):
                res, _ = self._get(f"/recommendations/{random_track_id}", params={"top_k": DEFAULT_TOP_K})
                
                if res.status_code != 200:
                    log_fail(f"Stress step failed at run {iteration}/{sample_size} for ID {random_track_id}: HTTP {res.status_code}")
                    return False
                
                payload = res.json()
                recs = payload.get("recommendations", [])
                if len(recs) != DEFAULT_TOP_K:
                    log_fail(f"Sizing error during stress run {iteration}: Expected {DEFAULT_TOP_K}, received {len(recs)}")
                    return False
                
                observed_set: Set[int] = set()
                
                for rec in recs:
                    if not self._validate_individual_track_item(rec) or not self._validate_vector_score_boundaries(rec):
                        log_fail(f"Malformed schema variables or data gap tracked at stress run {iteration}.")
                        return False
                    
                    r_id = int(rec["track_id"])
                    if r_id == random_track_id or r_id in observed_set:
                        log_fail(f"Deduplication or self-lookup anomaly tracked at stress loop step {iteration}.")
                        return False
                    
                    observed_set.add(r_id)
                
                all_observed_prediction_sets.append(observed_set)

            # Enterprise Matrix Diversity Verification Pass: Perform all-to-all unique pairwise intersection comparisons
            pairwise_overlaps: List[float] = []
            num_sets = len(all_observed_prediction_sets)
            
            for i in range(num_sets):
                for j in range(i + 1, num_sets):
                    set_a = all_observed_prediction_sets[i]
                    set_b = all_observed_prediction_sets[j]
                    
                    # Intersect set computation bounds
                    intersection_size = len(set_a.intersection(set_b))
                    max_possible_overlap = min(len(set_a), len(set_b))
                    
                    overlap_fraction = intersection_size / max_possible_overlap if max_possible_overlap > 0 else 0.0
                    pairwise_overlaps.append(overlap_fraction)

            # Handle isolated fallback triggers for solitary trace profiles safely
            if pairwise_overlaps:
                self._cached_avg_overlap_pct = (sum(pairwise_overlaps) / len(pairwise_overlaps)) * 100
                self._cached_max_overlap_pct = max(pairwise_overlaps) * 100
            else:
                self._cached_avg_overlap_pct = 0.0
                self._cached_max_overlap_pct = 0.0

            log_info(f" -> Computed Global Average Pairwise Overlap: {self._cached_avg_overlap_pct:.2f}%")
            log_info(f" -> Computed Global Maximum Pairwise Overlap: {self._cached_max_overlap_pct:.2f}%")

            # Evaluate system diversification matrix against maximum allowed similarity thresholds
            if self._cached_max_overlap_pct > (MAX_ALLOWED_DIVERSITY_OVERLAP_FRACTION * 100):
                log_fail(
                    f"Inference redundancy detected. Pairwise overlap ceiling broken: "
                    f"{self._cached_max_overlap_pct:.2f}% out of allowed {MAX_ALLOWED_DIVERSITY_OVERLAP_FRACTION * 100:.2f}%."
                )
                return False

            log_pass("Pairwise similarity indexes clear. Diversity and response variance are verified.")
            return True
        except Exception as err:
            logger.error(f"[!] Randomized fuzzing stress test pipeline triggered execution faults: {err}")
            log_fail("Stress and diversity evaluation suite crash.")
            return False

    def execute_validation_suite(self) -> bool:
        """Iterates through the test execution registry matrix and tracks outputs dynamically."""
        start_time = time.perf_counter()
        
        # Sequentially trigger test cases in an architecture-compliant layout order
        for name, test_callable in self.test_cases.items():
            try:
                self.test_results[name] = test_callable()
            except Exception as err:
                logger.error(f"Execution boundary crash during test case '{name}': {err}")
                self.test_results[name] = False

        total_time_ms = (time.perf_counter() - start_time) * 1000
        return self._print_formatted_summary(total_time_ms)

    def _print_formatted_summary(self, total_time_ms: float) -> bool:
        """Gathers step data, outputs the QA status framework table, and prints system diagnostics metadata."""
        print("\n" + "=" * 60)
        print("          CELEST RECOMMENDATION VALIDATION REPORT")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_count = 0
        failed_count = 0

        for name, status_val in self.test_results.items():
            leader_dots = "." * (40 - len(name))
            if status_val is True:
                status_string = COLOR_PASS
                passed_count += 1
            else:
                status_string = COLOR_FAIL
                failed_count += 1
                
            print(f" {name} {leader_dots} {status_string}")

        global_pass = (failed_count == 0)
        overall_string = COLOR_PASS if global_pass else COLOR_FAIL

        print("=" * 60)
        print(f" Total Tests : {total_tests}")
        print(f" Passed      : {passed_count}")
        print(f" Failed      : {failed_count}")
        print(f" Total Time  : {total_time_ms:.2f} ms")
        print("=" * 60)
        print(f" OVERALL RESULT: {overall_string}")
        print("=" * 60)
        
        # Extended Analytical Diagnostics Output Segment
        print("\n" + "=" * 60)
        print("                 CACHE & DIVERSITY DIAGNOSTICS")
        print("=" * 60)
        print(f" Cold Request Memory Latency ..... {self._cached_cold_display:.2f} ms")
        print(f" Average Warm Request Latency .... {self._cached_avg_warm_display:.2f} ms")
        print(f" Average Pairwise Track Overlap .. {self._cached_avg_overlap_pct:.2f}%")
        print(f" Maximum Pairwise Track Overlap .. {self._cached_max_overlap_pct:.2f}%")
        print("=" * 60)

        # System Information Block: Extends observability matrix details for cross-environment debugging traces
        print("\n" + "=" * 60)
        print("                      SYSTEM INFORMATION")
        print("=" * 60)
        print(f" Python Version .................. {platform.python_version()}")
        print(f" FastAPI Version ................. {fastapi.__version__}")
        print(f" Pydantic Version ................ {pydantic.__version__}")
        print(f" Requests Version ................ {requests.__version__}")
        print(f" Operating System ................ {platform.system()} {platform.release()} ({platform.machine()})")
        print(f" Validator Runtime ............... {total_time_ms / 1000:.4f} seconds")
        print(f" Generation Timestamp ............ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        return global_pass


if __name__ == "__main__":
    validator = CelestEngineValidator()
    all_passed = validator.execute_validation_suite()
    
    if all_passed:
        sys.exit(0)
    else:
        sys.exit(1)
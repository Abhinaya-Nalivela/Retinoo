"""
Rural Healthcare Screening Network Discrete-Event Simulator (SIH 26038).
Simulates rural tele-ophthalmology screening workflows serving 100,000+ patients/year.
Calculates queues, delays, resource utilizations, and identifies clinical bottlenecks.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import numpy as np


@dataclass
class SimulationConfig:
    annual_patient_target: int = 100000
    working_days_per_year: int = 250
    working_hours_per_day: float = 8.0
    
    # Facilities & Hardware
    num_healthcare_centres: int = 15
    num_cameras: int = 20
    img_acquisition_time_min: float = 5.0
    img_quality_fail_rate: float = 0.12
    recapture_time_min: float = 3.0
    
    # Telemedicine Network & AI
    img_size_mb: float = 8.0
    compression_ratio: float = 4.0
    net_bandwidth_mbps: float = 2.0  # Rural 3G/4G link
    ai_processing_time_sec: float = 1.5
    num_ai_servers: int = 4
    
    # Clinical Review
    referable_dr_fraction: float = 0.18  # ~18% require ophthalmologist review
    doctor_review_time_min: float = 8.0
    num_ophthalmologists: int = 3


class RuralScreeningSimulator:
    """
    Simulates patient throughput, network delays, AI inference batching,
    and specialist review queues across rural Primary Healthcare Centres (PHCs).
    """

    def __init__(self, config: SimulationConfig = None):
        self.cfg = config or SimulationConfig()

    def run(self, days_to_simulate: int = 30, include_scenarios: bool = True) -> Dict[str, Any]:
        cfg = self.cfg
        total_working_min_per_day = cfg.working_hours_per_day * 60.0
        
        # Target daily demand across all PHCs
        target_daily_patients = cfg.annual_patient_target / cfg.working_days_per_year
        arrival_rate_per_min = target_daily_patients / total_working_min_per_day
        
        # 1. Image Acquisition & Camera Capacity
        effective_acquisition_time = (
            cfg.img_acquisition_time_min + (cfg.img_quality_fail_rate * cfg.recapture_time_min)
        )
        camera_daily_capacity_per_unit = total_working_min_per_day / effective_acquisition_time
        total_camera_daily_capacity = camera_daily_capacity_per_unit * cfg.num_cameras
        
        # Patients captured per day
        daily_patients_arrived = target_daily_patients
        daily_captured = min(daily_patients_arrived, total_camera_daily_capacity)
        camera_utilization = min(1.0, daily_patients_arrived / (total_camera_daily_capacity + 1e-6))
        
        # 2. Quality assessment & Recaptures
        recaptures_per_day = daily_captured * cfg.img_quality_fail_rate
        unresolvable_rejects_per_day = daily_captured * 0.02  # ~2% unresolvable severe defects
        valid_screenings_per_day = daily_captured - unresolvable_rejects_per_day
        
        # 3. Transmission Delay
        compressed_size_mb = cfg.img_size_mb / cfg.compression_ratio
        compressed_size_mbit = compressed_size_mb * 8.0
        transfer_time_sec = compressed_size_mbit / max(0.1, cfg.net_bandwidth_mbps)
        
        # 4. AI Server Inference
        ai_time_min = (cfg.ai_processing_time_sec / 60.0)
        ai_daily_capacity = (total_working_min_per_day / ai_time_min) * cfg.num_ai_servers
        ai_utilization = min(1.0, valid_screenings_per_day / (ai_daily_capacity + 1e-6))
        
        # 5. Doctor Review
        referable_cases_per_day = valid_screenings_per_day * cfg.referable_dr_fraction
        doctor_daily_capacity = (total_working_min_per_day / cfg.doctor_review_time_min) * cfg.num_ophthalmologists
        doctor_utilization = min(1.0, referable_cases_per_day / (doctor_daily_capacity + 1e-6))
        
        # Queuing estimates (M/M/c approximation)
        completed_per_day = valid_screenings_per_day if doctor_utilization < 1.0 else (
            (valid_screenings_per_day * (1.0 - cfg.referable_dr_fraction)) + doctor_daily_capacity
        )
        annual_throughput = completed_per_day * cfg.working_days_per_year
        target_achievement_pct = (annual_throughput / cfg.annual_patient_target) * 100.0
        
        # Queue delays
        avg_camera_wait_min = (effective_acquisition_time * camera_utilization) / (2.0 * max(0.01, 1.0 - camera_utilization * 0.95))
        avg_doc_wait_min = (cfg.doctor_review_time_min * doctor_utilization) / (2.0 * max(0.01, 1.0 - doctor_utilization * 0.95))
        
        # Bottleneck detection
        utilizations = {
            "Fundus Cameras (Acquisition)": camera_utilization,
            "Network Bandwidth (Transmission)": min(1.0, transfer_time_sec / 30.0),
            "AI Compute Server": ai_utilization,
            "Ophthalmologist Review": doctor_utilization,
        }
        primary_bottleneck = max(utilizations, key=utilizations.get)
        
        # Generate daily trajectory simulation for plotting
        np.random.seed(42)
        days = list(range(1, days_to_simulate + 1))
        daily_arrivals = np.random.poisson(target_daily_patients, days_to_simulate)
        daily_completed = np.clip(daily_arrivals * (completed_per_day / target_daily_patients), 0, None)
        cumulative_patients = np.cumsum(daily_completed)
        doc_queue_trajectory = np.maximum(0, np.cumsum(daily_arrivals * cfg.referable_dr_fraction - doctor_daily_capacity))
        
        return {
            "config": asdict(cfg),
            "metrics": {
                "target_annual_patients": cfg.annual_patient_target,
                "projected_annual_throughput": int(annual_throughput),
                "target_achievement_pct": round(target_achievement_pct, 1),
                "daily_patient_demand": round(target_daily_patients, 1),
                "daily_completed_screenings": round(completed_per_day, 1),
                "daily_recaptures": round(recaptures_per_day, 1),
                "daily_unresolvable_rejects": round(unresolvable_rejects_per_day, 1),
                "daily_referable_cases": round(referable_cases_per_day, 1),
                "transmission_latency_sec": round(transfer_time_sec, 2),
                "avg_camera_wait_min": round(avg_camera_wait_min, 1),
                "avg_doctor_review_wait_min": round(avg_doc_wait_min, 1),
                "camera_utilization_pct": round(camera_utilization * 100, 1),
                "ai_utilization_pct": round(ai_utilization * 100, 1),
                "doctor_utilization_pct": round(doctor_utilization * 100, 1),
                "primary_bottleneck": primary_bottleneck,
            },
            "trajectories": {
                "days": days,
                "daily_arrivals": daily_arrivals.tolist(),
                "daily_completed": daily_completed.tolist(),
                "cumulative_patients": cumulative_patients.tolist(),
                "doc_queue_length": doc_queue_trajectory.tolist(),
            },
            "scenarios": self._run_comparative_scenarios() if include_scenarios else [],
        }

    def _run_comparative_scenarios(self) -> List[Dict[str, Any]]:
        """Runs preset policy and hardware scaling scenarios."""
        scenarios = []
        
        # Scenario 1: Baseline
        res_base = self.run(include_scenarios=False)
        scenarios.append({
            "name": "1. Baseline Deployment",
            "cameras": self.cfg.num_cameras,
            "doctors": self.cfg.num_ophthalmologists,
            "bandwidth": f"{self.cfg.net_bandwidth_mbps} Mbps",
            "annual_throughput": res_base["metrics"]["projected_annual_throughput"],
            "target_pct": res_base["metrics"]["target_achievement_pct"],
            "bottleneck": res_base["metrics"]["primary_bottleneck"],
        })
        
        # Scenario 2: Low Bandwidth (0.5 Mbps)
        cfg_low_bw = SimulationConfig(**asdict(self.cfg))
        cfg_low_bw.net_bandwidth_mbps = 0.5
        res_low_bw = RuralScreeningSimulator(cfg_low_bw).run(include_scenarios=False)
        scenarios.append({
            "name": "2. Low Bandwidth Remote Area (512 Kbps)",
            "cameras": cfg_low_bw.num_cameras,
            "doctors": cfg_low_bw.num_ophthalmologists,
            "bandwidth": "0.5 Mbps",
            "annual_throughput": res_low_bw["metrics"]["projected_annual_throughput"],
            "target_pct": res_low_bw["metrics"]["target_achievement_pct"],
            "bottleneck": res_low_bw["metrics"]["primary_bottleneck"],
        })
        
        # Scenario 3: Additional Cameras (+10 cameras)
        cfg_more_cam = SimulationConfig(**asdict(self.cfg))
        cfg_more_cam.num_cameras = 30
        res_more_cam = RuralScreeningSimulator(cfg_more_cam).run(include_scenarios=False)
        scenarios.append({
            "name": "3. Camera Scaling (30 Cameras)",
            "cameras": 30,
            "doctors": cfg_more_cam.num_ophthalmologists,
            "bandwidth": f"{cfg_more_cam.net_bandwidth_mbps} Mbps",
            "annual_throughput": res_more_cam["metrics"]["projected_annual_throughput"],
            "target_pct": res_more_cam["metrics"]["target_achievement_pct"],
            "bottleneck": res_more_cam["metrics"]["primary_bottleneck"],
        })
        
        # Scenario 4: High Demand (150,000 patients/year)
        cfg_high_dem = SimulationConfig(**asdict(self.cfg))
        cfg_high_dem.annual_patient_target = 150000
        cfg_high_dem.num_cameras = 32
        cfg_high_dem.num_ophthalmologists = 5
        res_high_dem = RuralScreeningSimulator(cfg_high_dem).run(include_scenarios=False)
        scenarios.append({
            "name": "4. Scaled Network (150k pts/yr Target)",
            "cameras": 32,
            "doctors": 5,
            "bandwidth": f"{cfg_high_dem.net_bandwidth_mbps} Mbps",
            "annual_throughput": res_high_dem["metrics"]["projected_annual_throughput"],
            "target_pct": res_high_dem["metrics"]["target_achievement_pct"],
            "bottleneck": res_high_dem["metrics"]["primary_bottleneck"],
        })
        
        return scenarios

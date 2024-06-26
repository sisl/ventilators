import csv
import os
import sys
import datetime
import time
from tqdm import tqdm
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Engine and data related imports
from pulse.engine.PulseEngine import PulseEngine
from pulse.cdm.engine import SEDataRequest, SEDataRequestManager

# Mechanical ventilator (incl control schemes) imports
from pulse.cdm.mechanical_ventilator import eSwitch, eDriverWaveform
from pulse.cdm.mechanical_ventilator_actions import SEMechanicalVentilatorConfiguration, \
                                                    SEMechanicalVentilatorContinuousPositiveAirwayPressure, \
                                                    SEMechanicalVentilatorPressureControl, \
                                                    SEMechanicalVentilatorVolumeControl, \
                                                    SEMechanicalVentilatorHold, \
                                                    SEMechanicalVentilatorLeak, \
                                                    eMechanicalVentilator_PressureControlMode, \
                                                    eMechanicalVentilator_VolumeControlMode

# Intubation 
from pulse.cdm.patient_actions import eIntubationType, SEIntubation

# Units
from pulse.cdm.scalars import FrequencyUnit, PressureUnit, PressureTimePerVolumeUnit, \
                              TimeUnit, VolumeUnit, VolumePerPressureUnit, VolumePerTimeUnit, \
                              LengthUnit, MassUnit

# Patient related imports
from pulse.cdm.patient import SEPatientConfiguration, eSex
from pulse.cdm.io.patient import serialize_patient_from_file
from pulse.cdm.patient_actions import SEAcuteRespiratoryDistressSyndromeExacerbation
from pulse.cdm.patient_actions import SEDyspnea
from pulse.cdm.patient_actions import SERespiratoryFatigue
from pulse.cdm.patient_actions import SEAsthmaAttack
#from pulse.cdm.physiology import eLungCompartment

def ventilate():
    print("\n\nBeginning ventilator simulation")
    
    # Create the engine
    pulse = PulseEngine()

    # The data we want to get back from the engine
    data_requests = [
        # Physiology data (actual oracle data from patient)
        SEDataRequest.create_physiology_request("RespirationRate", unit=FrequencyUnit.Per_min),
        SEDataRequest.create_physiology_request("TidalVolume", unit=VolumeUnit.mL),
        SEDataRequest.create_physiology_request("MeanAirwayPressure", unit=PressureUnit.cmH2O),
        SEDataRequest.create_physiology_request("HeartRate", unit=FrequencyUnit.Per_min),

        # From ventilator
        SEDataRequest.create_mechanical_ventilator_request("RespirationRate", unit=FrequencyUnit.Per_min),
        SEDataRequest.create_mechanical_ventilator_request("TidalVolume", unit=VolumeUnit.mL),
        SEDataRequest.create_mechanical_ventilator_request("MeanAirwayPressure", unit=PressureUnit.cmH2O),
        # etc02
        SEDataRequest.create_mechanical_ventilator_request("EndTidalCarbonDioxideFraction"),
        # PIP
        SEDataRequest.create_mechanical_ventilator_request("PeakInspiratoryPressure", unit=PressureUnit.cmH2O),
        # (lung) volume        
        SEDataRequest.create_mechanical_ventilator_request("TotalLungVolume", unit=VolumeUnit.L),
        # Flow (TODO inspiratory or expiratory, or both?)
        SEDataRequest.create_mechanical_ventilator_request("InspiratoryFlow", unit=VolumePerTimeUnit.L_Per_s),       

    ]

    # Configure where data will be written to
    data_mgr = SEDataRequestManager(data_requests)
    results_folder = "/mnt/results/"
    results_filepath = f"{results_folder}lungfish.csv"
    data_mgr.set_results_filename(results_filepath)
    
    # Load a default patient
    data_root_dir = "/mnt/data/"
    pc = SEPatientConfiguration()
    pc.set_data_root_dir(data_root_dir)
    pc.set_patient_file(f"{data_root_dir}/patients/StandardMale.json")

    # ARDS - must set before initialisation
    ards = pc.get_conditions().get_acute_respiratory_distress_syndrome()
    ards.get_left_lung_affected().set_value(0.8)
    ards.get_right_lung_affected().set_value(0.8)
    ards.get_severity().set_value(0.8)

    # Don't need to initialize if we serialize from file
    # https://gitlab.kitware.com/physiology/engine/-/blob/stable/src/python/pulse/howto/HowTo_AsthmaAttack.py#L3
    # Initialize the engine with our configuration
    if not pulse.initialize_engine(pc, None):
        print("Unable to load stabilize engine")
        return

    # Get default data at time 0s from the engine
    results = pulse.pull_data()
    data_mgr.to_console(results)

    # # Can also exacerbate ARDS
    # exacerbation = SEAcuteRespiratoryDistressSyndromeExacerbation()
    # exacerbation.set_comment("Patient's Acute Respiratory Distress Syndrome is exacerbated")
    # exacerbation.get_left_lung_affected().set_value(0.9)
    # exacerbation.get_right_lung_affected().set_value(0.9)
    # exacerbation.get_severity().set_value(0.9)
    # pulse.process_action(exacerbation)

    # # Dyspnea
    # dsypnea = SEDyspnea()
    # dsypnea.set_comment("Patient's dsypnea occurs")
    # dsypnea.get_severity().set_value(1)
    # pulse.process_action(dsypnea)

    # # Asthma
    # asthma_attack = SEAsthmaAttack()
    # asthma_attack.set_comment("Patient undergoes asthma attack")
    # asthma_attack.get_severity().set_value(0.4)
    # pulse.process_action(asthma_attack)

    # # Respiratory Fatigue
    # fatigue = SERespiratoryFatigue()
    # fatigue.set_comment("Patient undergoes respiratory fatigue")
    # fatigue.get_severity().set_value(0.666)
    # pulse.process_action(fatigue)

    # Advance after the disease action
    pulse.advance_time_s(60)
    # Get the values of the data you requested at this time
    results = pulse.pull_data()
    # And write it out to the console
    data_mgr.to_console(results)

    # We then get them on a ventilator
    # We'll use a volume control mode (spec. assist control, so no pressure support)
    # to simulate the patient needing much assistance
    vc_ac = SEMechanicalVentilatorVolumeControl()
    vc_ac.set_connection(eSwitch.On)
    vc_ac.set_mode(eMechanicalVentilator_VolumeControlMode.AssistedControl)
    # TODO investigate leaky ventilator
    # Recall that with VC-AC we need to specify:
    # V_t, the tidal volume
    vc_ac.get_tidal_volume().set_value(600.0, VolumeUnit.mL)
    # f, the respiration rate (1/num. breaths per minute)
    vc_ac.get_respiration_rate().set_value(12.0, FrequencyUnit.Per_min)
    # PEEP, the positive end-expiratory pressure
    vc_ac.get_positive_end_expired_pressure().set_value(5.0, PressureUnit.cmH2O)
    # FiO2, the fraction of inspired oxygen
    vc_ac.get_fraction_inspired_oxygen().set_value(0.21)
    # Flow rate, the MAXIMUM flow rate at which a tidal volume can be delivered
    vc_ac.get_flow().set_value(50.0, VolumePerTimeUnit.L_Per_min)
    # Ti, the inspiratory time (how long will the patient be at pressure peak)
    vc_ac.get_inspiratory_period().set_value(1.0, TimeUnit.s)
    
    # In pulse, when you apply a ventilator action, it will continue to be applied until
    # you apply a new ventilator action, or you apply a ventilator hold action
    pulse.process_action(vc_ac)

    # See how the patient reacts on ventilation
    for i in tqdm(range(2), desc="Advancing some minutes ahead on ventilation"):
        pulse.advance_time_s(60)
        # Get the values of the data you requested at this time
        results = pulse.pull_data()
        # And write it out to the console
        data_mgr.to_console(results)

    print(f"Simulation complete! File should be saved to '{results_filepath}'")

ventilate()
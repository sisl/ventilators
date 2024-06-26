# Distributed under the Apache License, Version 2.0.
# See accompanying NOTICE file for details.

from pulse.cdm.patient import SEPatientConfiguration
from pulse.cdm.patient_actions import SEAcuteRespiratoryDistressSyndromeExacerbation
from pulse.cdm.physiology import eLungCompartment
from pulse.engine.PulseEngine import PulseEngine

def HowTo_ARDS():
    pulse = PulseEngine()
    pulse.set_log_filename("/mnt/results/test_results/howto/HowTo_ARDS.py.log")
    pulse.log_to_console(True)

    pc = SEPatientConfiguration()

    data_root_dir = "/pulse/bin/"
    pc.set_data_root_dir(data_root_dir)
    pc.set_patient_file("./patients/StandardMale.json")
    ards = pc.get_conditions().get_acute_respiratory_distress_syndrome()
    ards.get_severity(eLungCompartment.LeftLung).set_value(0.2)
    ards.get_severity(eLungCompartment.RightLung).set_value(0.1)

    # Initialize the engine with our configuration
    # NOTE: No data requests are being provided, so Pulse will return the default vitals data
    if not pulse.initialize_engine(pc, None):
        print("Unable to load stabilize engine")
        return

    # Get some data from the engine
    results = pulse.pull_data()
    pulse.print_results()

    # Perform an action to exacerbate the initial condition state
    exacerbation = SEAcuteRespiratoryDistressSyndromeExacerbation()
    exacerbation.set_comment("Patient's Acute Respiratory Distress Syndrome is exacerbated")
    exacerbation.get_severity(eLungCompartment.LeftLung).set_value(0.4)
    exacerbation.get_severity(eLungCompartment.RightLung).set_value(0.2)
    pulse.process_action(exacerbation)

    # Advance some time and print out the vitals
    pulse.advance_time_s(30)
    results = pulse.pull_data()
    pulse.print_results()

HowTo_ARDS()

